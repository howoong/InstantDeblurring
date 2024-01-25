"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional, Sequence

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map
# from torch.utils.data._utils.collate import collate, default_collate_fn_map
from collate_my import collate, default_collate_fn_map

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import (
    accumulate_along_rays_,
    render_weight_from_density,
    rendering,
)

import torch.nn.functional as F

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]
MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
]
LLFF_SCENES = [
    "fern",
    "flower",
    "fortress",
    "horns",
    "leaves",
    "orchids",
    "room",
    "trex"
]

def foo(key, pad, patch_size):  # patch_size는 패드가 아직 안붙은 크기 # TODO function name
    if key == "center9":
        size = pad*2 + patch_size, pad*2 + patch_size, (0,0,0,0,0,0)    # 00LRTB
    elif key == "corner1":
        size = pad + patch_size, pad + patch_size, (0,0,pad, 0, pad, 0)
    elif key == "corner2":
        size = pad + patch_size, pad + patch_size, (0,0,0, pad, pad, 0)
    elif key == "corner3":
        size = pad + patch_size, pad + patch_size, (0,0,0, pad, 0, pad)
    elif key == "corner4":
        size = pad + patch_size, pad + patch_size, (0,0,pad, 0, 0, pad)
    elif key == "edge5":
        size = pad + patch_size, pad*2 + patch_size, (0,0,0, 0, pad, 0)
    elif key == "edge6":
        size = pad*2 + patch_size, pad + patch_size, (0,0,0, pad, 0, 0)
    elif key == "edge7":
        size = pad + patch_size, pad*2 + patch_size, (0,0,0, 0, 0, pad)
    elif key == "edge8":
        size = pad*2 + patch_size, pad + patch_size, (0,0,pad, 0, 0, 0)
    else:
        raise NotImplementedError
    return size 


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )
def render_image_with_occgrid_deblur(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    rays_info = None,
    pad: int = 6,
    patch_size: int = 8,
    pad_patch_size: int = 20,
    num_patches: int = 64,
    ksize: int = 13,
    noconv: int = 100,
    valid_lof: int = 400,
    focus_lv = None,
    img_list = None,
    gamma: int = -1,
    tonemapping = None,
    gt = None,
    patchwise_argmin: bool = True
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, t)
        else:
            sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]

    if rays_info is None:    # test
        # Tonemapping   
        colors = colors.clamp(min=0)
        if gamma > 0:
            colors = colors ** (1. / gamma)
        elif gamma == -1:
            colors = radiance_field.tonemapping(colors)

        colors = colors.clamp(0,1)
        return (
            colors.reshape((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
        )

    else:
        rgb_maps = []
        pos = 0
        for region, size in rays_info:
            cur_rgb_map = colors[pos: pos+size]
            if region != "center9":
                h, w, padding = foo(region, pad, patch_size) 
                cur_rgb_map = F.pad(cur_rgb_map.view(-1, h, w, 3), padding, 'reflect')
            else:
                cur_rgb_map = cur_rgb_map.view(-1, pad*2 + patch_size, pad*2 + patch_size, 3)
            rgb_maps.append(cur_rgb_map)
            pos += size
        pad_colors = torch.concat(rgb_maps) # 64,8,8,3  # 얘가 왜 음수가 되지 도대체?

        if valid_lof <= 0:
            colors = pad_colors[:,pad:-pad,pad:-pad,:]
        else:
            C = radiance_field.kernel_C
            kernel_type = radiance_field.kernel_type

            kernel = radiance_field.kernel[0][img_list]
            kernel = kernel.reshape(num_patches, -1, C, ksize*ksize)
            kernel = F.softmax(kernel, dim=-1)
            kernel = kernel.view(num_patches, -1, C, ksize, ksize)
            if C == 1:
                kernel = kernel.repeat(1,1,3,1,1)
            
            rgb_map = pad_colors.permute(0,3,1,2).reshape(1,-1,pad_patch_size,pad_patch_size)
            rgb_map = rgb_map.repeat(valid_lof, 1, 1, 1)
            rgb_map = rgb_map.view(1, -1, pad_patch_size, pad_patch_size)   # 1 LNC P P
            kernel = kernel.permute(1,0,2,3,4)  # lof,N,C,K,K
            kernel = kernel.reshape(-1,1,ksize,ksize) # N*C*lof,1,K,K
            blurred_img = F.conv2d(rgb_map, kernel, groups=rgb_map.shape[1]) # 1, N*C*lof,p,p
            blurred_img = blurred_img.view(-1, num_patches, 3, patch_size, patch_size).permute(0,1,3,4,2)  # lof,N,p,p,3    
            
            if kernel_type == "rand_GD":
                if noconv > 0:    # 뷰별로 다르면 일괄적으로 처리 불가능. 
                    top_rgb_map = pad_colors[None,:, pad : -pad, pad : -pad] # 1,N,p,p,3
                    top_rgb_map = top_rgb_map.repeat(noconv, 1, 1, 1, 1) # toff, N, psize, psize, 3
                    blurred_img = torch.concat((blurred_img, top_rgb_map))
                idx = focus_lv
                idx = idx.unsqueeze(-1).repeat(1,1,1,3).unsqueeze(0)    # 애초에 여기서 리핏을 하니까 첨부터 64 8 8 3이여도 문제 없음
                colors = torch.gather(blurred_img, 0, idx)
                colors = colors.view(-1, 3)
                colors = colors.clamp(min=0)

                # Tonemapping   
                if gamma > 0:
                    colors = colors ** (1. / gamma)
                elif gamma == -1:
                    colors = radiance_field.tonemapping(colors)
                else: 
                    colors = colors
                colors = colors.clamp(0,1)

            elif kernel_type == "argmin":  
                sharp_img = pad_colors[:,pad:-pad, pad:-pad].clamp(min=0)

                # Tonemapping
                if gamma > 0:
                    blurred_img = blurred_img ** (1. / gamma)
                    sharp_img = sharp_img ** (1. / gamma)
                elif gamma == -1:
                    blurred_img = radiance_field.tonemapping(blurred_img)
                    sharp_img = radiance_field.tonemapping(sharp_img)
                else: 
                    blurred_img = blurred_img
                    sharp_img = sharp_img
                blurred_img = blurred_img.clamp(0,1)
                sharp_img = sharp_img.clamp(0,1)

                # blurred_img = torch.concat([blurred_img, sharp_img[None]])  # LV+1, B, p, p, 3
                gt = gt.view(64,8,8,3)
                
                if patchwise_argmin:
                    lv_diff = (gt - blurred_img).abs().sum(dim=(2,3,4))
                    soft_idx = F.softmax(lv_diff, dim=0)
                    idx = torch.argmin(lv_diff, dim=0, keepdim=True)
                    hard_idx = torch.zeros_like(soft_idx, memory_format=torch.legacy_contiguous_format).scatter_(0, idx, 1.0)
                    ste_idx = (hard_idx - soft_idx).detach() + soft_idx
                    ste_idx = ste_idx.unsqueeze(-1).repeat(1,1,3)
                    colors = (blurred_img * ste_idx[...,None,None,:]).sum(0).view(-1,3)
                else:
                    lv_diff = (gt - blurred_img).abs().sum(dim=(4))   # LV+1, B, p, p
                    soft_idx = F.softmax(lv_diff, dim=0)
                    idx = torch.argmin(lv_diff, dim=0, keepdim=True)
                    hard_idx = torch.zeros_like(soft_idx, memory_format=torch.legacy_contiguous_format).scatter_(0, idx, 1.0)
                    ste_idx = (hard_idx - soft_idx).detach() + soft_idx
                    import pdb; pdb.set_trace()
                    ste_idx = ste_idx.unsqueeze(-1).repeat(1,1,3)
                    colors = (blurred_img * ste_idx[...,None,None,:]).sum(0).view(-1,3)

            elif kernel_type == "flexible":
                blurred_imgs = []
                print(radiance_field.kn)
                # 당연히 안됨 쉐입이 다를 수 있으니... 
                # 그러면 kn으로 나눠서 해야됨.. 근데 또 그러면 포커스레벨이 애매해짐;;
                import pdb; pdb.set_trace()
                for i in range(radiance_field.kn):
                    cur_kernel = radiance_field.kernel[i]   # 이미지별로 다르텐데... Img, lv, 3, k, k라고 가정 ㄴㄴ 가정자체를 못하지;
                    # 뷰별로 레벨별 픽셀 수가 다름. 1번 이미지는 3사이즈 짜리가 10개, 2번 이미지는 20개 이러면 하나의 텐서로 못나타내지
                    # 즉 크기가 다른 컨볼루션을 동시에 해야하는데 -> 근데 최종적으로는 N개의 커널로 변환할테니 스택같은거만 잘 하면?
                    '''
                    41, 400, 3, 13, 13임 원래는. 근데 이제 이걸 뷰별로 커널 크기를 자유롭게 가져가려 함. 그러면 K크기의 커널에 할당되는 레벨의
                    수가 최대 41*400이 될 수도 있음. 
                    ex. 크기가 13짜리 커널에 해당되는 애들
                    1번이미지: 20, 3, 13,13
                    2번이미지: 17, 3, 13, 13
                    3번이미지: 11, 3, 13, 13
                    ...
                    41번이미:  7,  3, 13, 13
                    근데 이 중에서 64개 이미지 골라서 걔네 커널  다 합치면 800, 3, 13, 13이라고 가정. 즉 레벨이 800개. 
                    그러면 이미지도 복사해서 800개로 만들고 컨볼하면 800, 8, 8, 3. 원래는 64, N, 8, 8, 3이였음
                    이미지는 64 20 20 3, focus_lv는 64 8 8 3
                    다른 것들도 다 해서 스택까지 하면 16400, 8, 8, 3. 이상태에서 게더를 어떻게?
                    가장 나이브한건 이미지별로 모아서 하나하나 스택해서 41, 400을 만들어 주는 것
                    아니면 이제는 그냥 첨부터 필요한애만 쓰면? 
                     -> 일단 unfold로 나눠버리면 패치를 다 쪼갤 수 있고, 커널을 잘 골라야 함. 
                     -> 애초에 레벨을 그러면 1번째 이미지는 0~399, 1번째 이미지는 400더해서 400~799로 만들고 단순 인덱싱하면
                     가능하긴 ㅏㅎ네!!!! 그러면 게더 없이
                    커널을 그냥 파라미터리스트 41*400으로 만들어서 인덱싱해서 스택하면 되긴하네?
                    근데 게더는?
                    '''
                    n = cur_kernel.shape[0] # 여기서 말하는 n은 아마 레벨 개수일 것. 근데 나느 이미지가 앞에 꼈으니...
                    if n == 0:
                        continue
                    if i == 0:  # 0이면 샤프
                        cur_rgb_map = pad_colors
                    else:
                        cur_rgb_map = pad_colors[:, pad-(6-i) : -(pad-(6-i)), pad-(6-i) : -(pad-(6-i))]    # 64 P,P,3
                    p = cur_rgb_map.shape[1]
                    cur_rgb_map = cur_rgb_map.permute(0,3,1,2).reshape(1,-1,p,p)    # 1 BC, P, P
                    cur_rgb_map = cur_rgb_map.repeat(n, 1, 1, 1) # lv, BC, P, P  이것도 아님 사용되는 "유니크"커널 개수만큼만 잇으면 됨
                    cur_rgb_map = cur_rgb_map.view(1, -1, p, p)   # 1 lv*B*C P P
                    k = cur_kernel.shape[-1]
                    cur_kernel = cur_kernel.view(n, radiance_field.kernel_C, k*k)
                    cur_kernel = F.softmax(cur_kernel, dim=-1)
                    cur_kernel = cur_kernel.view(n, radiance_field.kernel_C, k, k).repeat(num_patches, 1, 1, 1, 1)   # B,lv,C,3,3
                    cur_kernel = cur_kernel.permute(1,0,2,3,4)  # lv,B,C,K,K
                    cur_kernel = cur_kernel.reshape(-1,1,k,k) # lv*B*C,1,K,K
                    blurred_img = F.conv2d(cur_rgb_map, cur_kernel, groups=cur_rgb_map.shape[1]) # 1, lv*B*C,p,p
                    blurred_img = blurred_img.view(n, num_patches, 3, patch_size, patch_size).permute(0,1,3,4,2)  # lv,B,p,p,3 
                    blurred_imgs.append(blurred_img)

                blurred_imgs.append(pad_rgb_map[:,pad:-pad, pad:-pad].repeat(self.noconv,1,1,1,1))  # 이것도 이미지별로 달를텐데..
                blurred_img = torch.concat(blurred_imgs)    # lof, B, psize, psize, 3

                idx = focus_lv.unsqueeze(-1).repeat(1,1,1,3).unsqueeze(0)
                rgb_map = torch.gather(blurred_img, 0, idx)
                rgb_map = rgb_map.view(-1, 3)   

        return (
            colors.reshape((4096, -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
        )


def render_image_with_propnet(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    estimator: PropNetEstimator,
    rays: Rays,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            prop_samples=num_samples_per_prop,
            num_samples=num_samples,
            n_rays=chunk_rays.origins.shape[0],
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            stratified=radiance_field.training,
            requires_grad=proposal_requires_grad,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        extras,
    )


def render_image_with_propnet_deblur(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    estimator: PropNetEstimator,
    rays: Rays,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
    rays_info = None,
    pad: int = 6,
    patch_size: int = 8
):
    """Render the pixels of an image."""
    # 일단은 다 렌더링하고 가운데만 뽑는걸로 해보자 즉 블러링 없이 패치와이즈로만 한다 했을 때 얼마나 시간이 걸리는지. 
    # 그리고 뭔가 그냥 0~392중에서 뽑게하고 자연스럽게 패딩 붙이는 것도 괜찮을 것 같은데.. 사이즈 안맞는 만큼 리플렉트 패딩?이게 쉐입 측면에서 가능한건가근데?
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            prop_samples=num_samples_per_prop,
            num_samples=num_samples,
            n_rays=chunk_rays.origins.shape[0],
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            stratified=radiance_field.training,
            requires_grad=proposal_requires_grad,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    # what is collate?
    # 이제 ray_info보고 다시 for문 돌아야함;;;
    if rays_info is not None:
        rgb_maps = []
        pos = 0
        for region, size in rays_info:
            cur_rgb_map = colors[pos: pos+size]
            if region != "center9":
                h, w, padding = foo(region, pad, patch_size) 
                cur_rgb_map = F.pad(cur_rgb_map.view(-1, h, w, 3), padding, 'reflect')
            else:
                cur_rgb_map = cur_rgb_map.view(-1, pad*2 + patch_size, pad*2 + patch_size, 3)
            rgb_maps.append(cur_rgb_map)
            pos += size
        pad_colors = torch.concat(rgb_maps)
        colors = pad_colors[:,pad:-pad, pad:-pad, :]
        return (
            colors.reshape((4096, -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            extras,
        )
    else:
        return (
            colors.reshape((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            extras,
        )

def render_image_with_propnet_deblur_real(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    estimator: PropNetEstimator,
    rays: Rays,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
    rays_info = None,
    pad: int = 6,
    patch_size: int = 8,
    pad_patch_size: int = 20,
    num_patches: int = 64,
    ksize: int = 13,
    noconv: int = 100,
    valid_lof: int = 400,
    focus_lv = None,
    img_list = None,
    gamma: int = -1,
    tonemapping = None,
    gt = None,
    patchwise_argmin: bool = True

):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            prop_samples=num_samples_per_prop,
            num_samples=num_samples,
            n_rays=chunk_rays.origins.shape[0],
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            stratified=radiance_field.training,
            requires_grad=proposal_requires_grad,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    # what is collate?
    # 이제 ray_info보고 다시 for문 돌아야함;;;
    if rays_info is None:    # test
        # Tonemapping   
        colors = colors.clamp(min=0)
        if gamma > 0:
            colors = colors ** (1. / gamma)
        elif gamma == -1:
            colors = radiance_field.tonemapping(colors)

        # clamp를 해야하나 말아야하나?
        colors = colors.clamp(0,1)

        return (
            colors.reshape((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            extras,
        )
    else:
        rgb_maps = []
        pos = 0
        for region, size in rays_info:
            cur_rgb_map = colors[pos: pos+size]
            if region != "center9":
                h, w, padding = foo(region, pad, patch_size) 
                cur_rgb_map = F.pad(cur_rgb_map.view(-1, h, w, 3), padding, 'reflect')
            else:
                cur_rgb_map = cur_rgb_map.view(-1, pad*2 + patch_size, pad*2 + patch_size, 3)
            rgb_maps.append(cur_rgb_map)
            pos += size
        pad_colors = torch.concat(rgb_maps) # 64,8,8,3  # 얘가 왜 음수가 되지 도대체?

        if valid_lof <= 0:
            colors = pad_colors[:,pad:-pad,pad:-pad,:]
        else:
            C = radiance_field.kernel_C
            kernel_type = radiance_field.kernel_type

            kernel = radiance_field.kernel[0][img_list]
            kernel = kernel.reshape(num_patches, -1, C, ksize*ksize)
            kernel = F.softmax(kernel, dim=-1)
            kernel = kernel.view(num_patches, -1, C, ksize, ksize)
            if C == 1:
                kernel = kernel.repeat(1,1,3,1,1)
            
            rgb_map = pad_colors.permute(0,3,1,2).reshape(1,-1,pad_patch_size,pad_patch_size)
            rgb_map = rgb_map.repeat(valid_lof, 1, 1, 1)
            rgb_map = rgb_map.view(1, -1, pad_patch_size, pad_patch_size)   # 1 LNC P P
            kernel = kernel.permute(1,0,2,3,4)  # lof,N,C,K,K
            kernel = kernel.reshape(-1,1,ksize,ksize) # N*C*lof,1,K,K
            blurred_img = F.conv2d(rgb_map, kernel, groups=rgb_map.shape[1]) # 1, N*C*lof,p,p
            blurred_img = blurred_img.view(-1, num_patches, 3, patch_size, patch_size).permute(0,1,3,4,2)  # lof,N,p,p,3    
            
            if kernel_type in ["rand_GD", "flexible"]:
                if noconv > 0:    # 뷰별로 다르면 일괄적으로 처리 불가능. 
                    top_rgb_map = pad_colors[None,:, pad : -pad, pad : -pad] # 1,N,p,p,3
                    top_rgb_map = top_rgb_map.repeat(noconv, 1, 1, 1, 1) # toff, N, psize, psize, 3
                    blurred_img = torch.concat((blurred_img, top_rgb_map))
                idx = focus_lv
                idx = idx.unsqueeze(-1).repeat(1,1,1,3).unsqueeze(0)    # 애초에 여기서 리핏을 하니까 첨부터 64 8 8 3이여도 문제 없음
                colors = torch.gather(blurred_img, 0, idx)
                colors = colors.view(-1, 3)
                colors = colors.clamp(min=0)

                # Tonemapping   
                if gamma > 0:
                    colors = colors ** (1. / gamma)
                elif gamma == -1:
                    colors = radiance_field.tonemapping(colors)
                else: 
                    colors = colors
                colors = colors.clamp(0,1)

            elif kernel_type == "argmin":  
                sharp_img = pad_colors[:,pad:-pad, pad:-pad].clamp(min=0)

                # Tonemapping
                if gamma > 0:
                    blurred_img = blurred_img ** (1. / gamma)
                    sharp_img = sharp_img ** (1. / gamma)
                elif gamma == -1:
                    blurred_img = radiance_field.tonemapping(blurred_img)
                    sharp_img = radiance_field.tonemapping(sharp_img)
                else: 
                    blurred_img = blurred_img
                    sharp_img = sharp_img
                blurred_img = blurred_img.clamp(0,1)
                sharp_img = sharp_img.clamp(0,1)

                # blurred_img = torch.concat([blurred_img, sharp_img[None]])  # LV+1, B, p, p, 3
                gt = gt.view(64,8,8,3)
                
                if patchwise_argmin:
                    lv_diff = (gt - blurred_img).abs().sum(dim=(2,3,4))
                    soft_idx = F.softmax(lv_diff, dim=0)
                    idx = torch.argmin(lv_diff, dim=0, keepdim=True)
                    hard_idx = torch.zeros_like(soft_idx, memory_format=torch.legacy_contiguous_format).scatter_(0, idx, 1.0)
                    ste_idx = (hard_idx - soft_idx).detach() + soft_idx
                    ste_idx = ste_idx.unsqueeze(-1).repeat(1,1,3)
                    colors = (blurred_img * ste_idx[...,None,None,:]).sum(0).view(-1,3)
                else:
                    lv_diff = (gt - blurred_img).abs().sum(dim=(4))   # LV+1, B, p, p
                    soft_idx = F.softmax(lv_diff, dim=0)
                    idx = torch.argmin(lv_diff, dim=0, keepdim=True)
                    hard_idx = torch.zeros_like(soft_idx, memory_format=torch.legacy_contiguous_format).scatter_(0, idx, 1.0)
                    ste_idx = (hard_idx - soft_idx).detach() + soft_idx
                    import pdb; pdb.set_trace()
                    ste_idx = ste_idx.unsqueeze(-1).repeat(1,1,3)
                    colors = (blurred_img * ste_idx[...,None,None,:]).sum(0).view(-1,3)

            # elif kernel_type == "flexible":
            #     blurred_imgs = []
            #     import pdb; pdb.set_trace()
            #     for i in range(radiance_field.kn):
            #         cur_kernel = radiance_field.kernel[i]   # 이미지별로 다르텐데... Img, lv, 3, k, k라고 가정 ㄴㄴ 가정자체를 못하지;
            #         n = cur_kernel.shape[0] # 여기서 말하는 n은 아마 레벨 개수일 것. 근데 나느 이미지가 앞에 꼈으니...
            #         if n == 0:
            #             continue
            #         if i == 0:  # 0이면 샤프
            #             cur_rgb_map = pad_colors
            #         else:
            #             cur_rgb_map = pad_colors[:, pad-(6-i) : -(pad-(6-i)), pad-(6-i) : -(pad-(6-i))]    # 64 P,P,3
            #         p = cur_rgb_map.shape[1]
            #         cur_rgb_map = cur_rgb_map.permute(0,3,1,2).reshape(1,-1,p,p)    # 1 BC, P, P
            #         cur_rgb_map = cur_rgb_map.repeat(n, 1, 1, 1) # lv, BC, P, P  이것도 아님 사용되는 "유니크"커널 개수만큼만 잇으면 됨
            #         cur_rgb_map = cur_rgb_map.view(1, -1, p, p)   # 1 lv*B*C P P
            #         k = cur_kernel.shape[-1]
            #         cur_kernel = cur_kernel.view(n, radiance_field.kernel_C, k*k)
            #         cur_kernel = F.softmax(cur_kernel, dim=-1)
            #         cur_kernel = cur_kernel.view(n, radiance_field.kernel_C, k, k).repeat(num_patches, 1, 1, 1, 1)   # B,lv,C,3,3
            #         cur_kernel = cur_kernel.permute(1,0,2,3,4)  # lv,B,C,K,K
            #         cur_kernel = cur_kernel.reshape(-1,1,k,k) # lv*B*C,1,K,K
            #         blurred_img = F.conv2d(cur_rgb_map, cur_kernel, groups=cur_rgb_map.shape[1]) # 1, lv*B*C,p,p
            #         blurred_img = blurred_img.view(n, num_patches, 3, patch_size, patch_size).permute(0,1,3,4,2)  # lv,B,p,p,3 
            #         blurred_imgs.append(blurred_img)

            #     blurred_imgs.append(pad_rgb_map[:,pad:-pad, pad:-pad].repeat(self.noconv,1,1,1,1))  # 이것도 이미지별로 달를텐데..
            #     blurred_img = torch.concat(blurred_imgs)    # lof, B, psize, psize, 3

            #     idx = focus_lv.unsqueeze(-1).repeat(1,1,1,3).unsqueeze(0)
            #     rgb_map = torch.gather(blurred_img, 0, idx)
            #     rgb_map = rgb_map.view(-1, 3)   

        return (
            colors.reshape((4096, -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            extras,
        )

@torch.no_grad()
def render_image_with_occgrid_test(
    max_samples: int,
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    gamma: float = 2.2
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = (
            t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0
        )
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    device = rays.origins.device
    opacity = torch.zeros(num_rays, 1, device=device)
    depth = torch.zeros(num_rays, 1, device=device)
    rgb = torch.zeros(num_rays, 3, device=device)

    ray_mask = torch.ones(num_rays, device=device).bool()

    # 1 for synthetic scenes, 4 for real scenes
    min_samples = 1 if cone_angle == 0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(
            0, n_grids * 2, device=t_mins.device, dtype=torch.int64
        ).expand(num_rays, n_grids * 2)

    opc_thre = 1 - early_stop_eps

    while iter_samples < max_samples:

        n_alive = ray_mask.sum().item()
        if n_alive == 0:
            break

        # the number of samples to add on each ray
        n_samples = max(min(num_rays // n_alive, 64), min_samples)
        iter_samples += n_samples

        # ray marching
        (intervals, samples, termination_planes) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            n_samples,
            True,
            ray_mask,
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # get rgb and sigma from radiance field
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # volume rendering using native cuda scan
        weights, _, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=num_rays,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        if alpha_thre > 0:
            vis_mask = alphas >= alpha_thre
            ray_indices, rgbs, weights, t_starts, t_ends = (
                ray_indices[vis_mask],
                rgbs[vis_mask],
                weights[vis_mask],
                t_starts[vis_mask],
                t_ends[vis_mask],
            )

        accumulate_along_rays_(
            weights,
            values=rgbs,
            ray_indices=ray_indices,
            outputs=rgb,
        )
        accumulate_along_rays_(
            weights,
            values=None,
            ray_indices=ray_indices,
            outputs=opacity,
        )
        accumulate_along_rays_(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            outputs=depth,
        )
        # update near_planes using termination planes
        near_planes = termination_planes
        # update rays status
        ray_mask = torch.logical_and(
            # early stopping
            opacity.view(-1) <= opc_thre,
            # remove rays that have reached the far plane
            packed_info[:, 1] == n_samples,
        )
        total_samples += ray_indices.shape[0]

    rgb = rgb + render_bkgd * (1.0 - opacity)
    depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)

    return (
        rgb.view((*rays_shape[:-1], -1)),
        opacity.view((*rays_shape[:-1], -1)),
        depth.view((*rays_shape[:-1], -1)),
        total_samples,
    )
