"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

# import imageio.v2 as imageio
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import glob
from kornia import create_meshgrid
from PIL import Image
from torchvision import transforms as T

from .utils import Rays

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def get_ray_directions_blender(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]+0.5
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -torch.ones_like(i)],
                             -1)  # (H, W, 3)

    return directions


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def ndc_rays_blender(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def _load_renderings(root_fp: str, subject_id: str, split: str, hold_every:int, downsample:int):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    # down sampling, 테스트셋 분리
    data_dir = os.path.join(root_fp, subject_id)
    poses_bounds = np.load(os.path.join(data_dir, 'poses_bounds.npy'))  # (N_images, 17)
    image_paths = sorted(glob.glob(os.path.join(data_dir, f'images_{downsample}/*')))
    blender2opencv = np.eye(4)

    # load full resolution image then resize
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    near_fars = poses_bounds[:, -2:]  # (N_images, 2)
    hwf = poses[:, :, -1]

    # Step 1: rescale focal length according to training resolution
    H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
    img_wh = np.array([int(W / downsample), int(H / downsample)])
    focal = [focal * img_wh[0] / W, focal * img_wh[1] / H]

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # (N_images, 3, 4) exclude H, W, focal
    poses, pose_avg = center_poses(poses, blender2opencv)

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = near_fars.min()
    scale_factor = near_original * 0.75  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    near_fars /= scale_factor
    poses[..., 3] /= scale_factor

    # build rendering path
    # N_views, N_rots = 120, 2
    # tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    # up = normalize(poses[:, :3, 1].sum(0))
    # rads = np.percentile(np.abs(tt), 90, 0)
    # render_path = get_spiral(poses, near_fars, N_views=N_views)

    # distances_from_center = np.linalg.norm(poses[..., 3], axis=1)
    # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
    # center image

    # ray directions for all pixels, same for all images (same H, W, focal)
    W, H = img_wh
    directions = get_ray_directions_blender(H, W, focal)  # (H, W, 3)

    average_pose = average_poses(poses)
    dists = np.sum(np.square(average_pose[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.arange(0, poses.shape[0], hold_every)  # [np.argmin(dists)]
    img_list = i_test if split != 'train' else list(set(np.arange(len(poses))) - set(i_test))

    img_list = img_list
    # use first N_images-1 to train, the LAST is val
    all_rays = []
    all_rgbs = []
    transform = T.ToTensor()

    for i in img_list:
        image_path = image_paths[i]

        c2w = torch.FloatTensor(poses[i])

        img = Image.open(image_path).convert('RGB')
        if downsample != 1.0:
            img = img.resize(img_wh, Image.LANCZOS)
        img = transform(img)  # (3, h, w)

        img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
        all_rgbs += [img]
        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)

        rays_o, rays_d = ndc_rays_blender(H, W, focal[0], 1.0, rays_o, rays_d)

        all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

    all_rays = torch.stack(all_rays, 0).reshape(-1,*img_wh[::-1], 6)   # (len(meta['frames]),h,w, 3)
    all_rgbs = torch.stack(all_rgbs, 0).reshape(-1,*img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
    N = all_rays.shape[0]
    return all_rgbs, all_rays, N


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "cake",
        "caps",
        "cisco",
        "coral",
        "cupcake",
        "cups",
        "daisy",
        "seal",
        "tools"
    ]

    WIDTH, HEIGHT = 600, 400       # LLFF
    NEAR, FAR = 0.0, 1.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        num_patches: int = 64,
        patch_size: int = 8,
        pad: int = 6
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        # assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        print(self.near, self.far)
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.device = device
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.pad = pad
        self.pad_patch_size = patch_size + self.pad*2

        # 데이터셋보고 정해야됨
        data_type = root_fp.split("/")[-1]
        if data_type.split("_")[1] == "camera":
            prefix = "blur"
        elif data_type.split("_")[1] == "defocus":
            prefix = "defocus"
        else:
            print("Unsupported dataset")
            raise NotImplementedError
        subject_id = prefix + subject_id

        if data_type.split("_")[0] == "synthetic":
            downsample = 1
            hold_every = 8
        elif data_type.split("_")[0] == "real":
            downsample = 4
            data_dir = os.path.join(root_fp, subject_id)
            filelist = os.listdir(data_dir)
            for f in filelist:
                if f.startswith("hold"):
                    hold_every = int(f.split("=")[-1])
                    print(hold_every)
        else:
            print("Unsupported dataset")
            raise NotImplementedError
        
        self.rgbs, self.rays, self.N = _load_renderings(
            root_fp, subject_id, split, hold_every, downsample
        )
        self.generate_patches()


    def generate_patches(self):
        self.min_w = self.WIDTH // self.patch_size
        self.min_h = self.HEIGHT // self.patch_size
        pad = self.pad
        cor_size = (self.patch_size + pad) ** 2
        edge_size = (self.patch_size + pad) * (self.patch_size + pad*2) 
        center_size = (self.patch_size + pad*2) * (self.patch_size + pad*2) 
        self.windows = []
        self.windows_gt = []

        for h in range(self.min_h):
            for w in range(self.min_w):
                if h == 0:
                    if w == 0:  # TL corner
                        self.windows.append([[0, self.patch_size + pad, 0, self.patch_size + pad], ["corner1", cor_size]])    # TL
                        self.windows_gt.append([0, self.patch_size, 0, self.patch_size])    # TL
                    elif w == self.min_w - 1: # TR corner
                        self.windows.append([[self.WIDTH - self.patch_size - pad, self.WIDTH, 0, self.patch_size + pad], ["corner2", cor_size]]) # TR
                        self.windows_gt.append([self.WIDTH - self.patch_size, self.WIDTH, 0, self.patch_size]) # TR
                    else: # T edge
                        self.windows.append([[w*self.patch_size - pad, (w+1)*self.patch_size + pad, 0, self.patch_size + pad], ["edge5", edge_size]])
                        self.windows_gt.append([w*self.patch_size, (w+1)*self.patch_size, 0, self.patch_size])
                elif h == self.min_h -1:
                    if w == 0: # BL corner
                        self.windows.append([[0, self.patch_size + pad, self.HEIGHT - self.patch_size - pad, self.HEIGHT], ["corner4", cor_size]])    # BL
                        self.windows_gt.append([0, self.patch_size, self.HEIGHT - self.patch_size, self.HEIGHT])    # BL
                    elif w == self.min_w - 1:    # BR corner
                        self.windows.append([[self.WIDTH - self.patch_size - pad, self.WIDTH, self.HEIGHT - self.patch_size - pad, self.HEIGHT], ["corner3", cor_size]]) # BR
                        self.windows_gt.append([self.WIDTH - self.patch_size, self.WIDTH, self.HEIGHT - self.patch_size, self.HEIGHT]) # BR
                    else: # B_edge
                        self.windows.append([[w*self.patch_size - pad, (w+1)*self.patch_size + pad, self.HEIGHT - self.patch_size - pad, self.HEIGHT], ["edge7", edge_size]])
                        self.windows_gt.append([w*self.patch_size, (w+1)*self.patch_size, self.HEIGHT - self.patch_size, self.HEIGHT])
                elif w == 0:    # L edge
                    self.windows.append([[0, self.patch_size + pad, h * self.patch_size - pad, (h+1)*self.patch_size + pad], ["edge8", edge_size]])
                    self.windows_gt.append([0, self.patch_size, h * self.patch_size, (h+1)*self.patch_size])
                elif w == self.min_w - 1 :   # R edge
                    self.windows.append([[self.WIDTH - self.patch_size - pad, self.WIDTH, h * self.patch_size - pad, (h+1)*self.patch_size + pad], ["edge6", edge_size]])
                    self.windows_gt.append([self.WIDTH - self.patch_size, self.WIDTH, h * self.patch_size, (h+1)*self.patch_size])
                else:   # center
                    self.windows.append([[w*self.patch_size - pad, (w+1)*self.patch_size + pad, h*self.patch_size - pad, (h+1)*self.patch_size + pad], ["center9", center_size]])
                    self.windows_gt.append([w*self.patch_size, (w+1)*self.patch_size, h*self.patch_size, (h+1)*self.patch_size])
        
        self.num_windows = len(self.windows)

    def __len__(self):
        return self.N

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]

        '''
        TODO
        360은 블랙으로 줬음 신테틱은 화이트. 근데 테스트할대는 360도 화이트로함
        근데 llff는 rgba가 아니라 rgb라 3채널임. 근데 360도 3채널인데?
        근데 찍어보면 랜덤이라고 뜸 360 ㅋㅋㅋㅅㅂ 일단은 래덤으로
        
        '''
        if self.training:   
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.device)

        # TODO 360에서 생략함
        # pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            # "color_bkgd": None,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays, num_patches, patch_size, pad_patch_size):
        self.num_rays = num_rays
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.pad_patch_size = pad_patch_size
        self.w_bound = self.W - self.patch_size
        self.h_bound = self.H - self.patch_size

    def fetch_data(self, index):        # 패치 구현.. 그리고 프로포절넷 2개에 유니폼도 해보기!!! occ도 해봐야되고!!
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays
        num_patches = self.num_patches
        patch_size = self.patch_size
        pad_patch_size = self.pad_patch_size

        if self.training:
            image_id = torch.randint(
                0,
                self.N,
                size=(num_patches,),
                device=self.rgbs.device,    #cuda
            )

            # 그냥 기존에 하던 방식으로 이러면 근데 슬라이싱을 못하는데.. 일단은 포문..?
            rgb_train = []
            rays_train = []
            rays_info = []
            chunk = 0
            patch_id = torch.randint(0, self.num_windows, size=(num_patches,), device=self.rgbs.device)
            for iid,pid in zip(image_id, patch_id):    # 매번 이걸 돌아야 한다는게.. 일단은 시간비교 해봐야함
                rays, gt = self.windows[pid], self.windows_gt[pid]
                coord, ray_info = rays
                w, ww, h, hh = coord 
                chunk += ray_info[1]
                rays_info.append(ray_info)
                rays_train.append(self.rays[iid, h:hh, w:ww].reshape(-1,6))
                
                w, ww, h, hh = gt
                rgb_train.append(self.rgbs[iid, h:hh, w:ww])

            rgb = torch.stack(rgb_train).reshape(-1,3).to(self.device)
            rays = torch.concat(rays_train).to(self.device)

        else:   # 이거는 안고쳐도 될듯. 어차피 따로 처리하니까. 근데 ray만드는 부분 좀 건드려야함
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.rgbs.device),
                torch.arange(self.HEIGHT, device=self.rgbs.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
            rgb = self.rgbs[image_id, y, x].to(self.device)  # (num_rays, 3)
            rays = self.rays[image_id, y, x].to(self.device)
            rays_info = None

        origins = rays[..., :3]
        viewdirs = rays[..., 3:]

        if not self.training:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgb = torch.reshape(rgb, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "rays_info": rays_info
        }
