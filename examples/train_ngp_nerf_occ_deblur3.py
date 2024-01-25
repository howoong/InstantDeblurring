"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField

# from examples.utils import (
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid_deblur,
    render_image_with_occgrid_test,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    help="which scene to use",
)

parser.add_argument(
    "--tag",
    type=str,
    default="",
)

parser.add_argument(
    "--type",
    type=str,
    default="mine",
)
parser.add_argument(
    "--wdecay",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--lof",
    type=int,
    default=0,
)
parser.add_argument(
    "--noconv",
    type=int,
    default=0,
)
parser.add_argument(
    "--ksize",
    type=int,
    default=13,
)
parser.add_argument(
    "--num_patches",
    type=int,
    default=64,
)
parser.add_argument(
    "--patch_size",
    type=int,
    default=8,
)
parser.add_argument(
    "--fmo",
    type=str,
    default="AIF",
)
parser.add_argument(
    "--dataset_tag",
    type=str,
    default=None,
)
parser.add_argument(
    "--grouping",
    type=str,
    default="quantile",
)
parser.add_argument(
    "--iteration",
    type=int,
    default=20000,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4096,
)
parser.add_argument(
    "--gamma",
    type=float,
    default=2.2,
)
parser.add_argument(
    "--lr_kernel",
    type=float,
    default=1e-3,
)
parser.add_argument(
    "--lr_ngp",
    type=float,
    default=1e-2,
)
parser.add_argument(
    "--lr_prop",
    type=float,
    default=-1.0,
)
parser.add_argument(
    "--lr_kernel_decay_target_ratio",
    type=float,
    default=-1
)
parser.add_argument(
    "--render_train",
    type=int,
    default=0
)
parser.add_argument(
    "--vis_kernel",
    type=int,
    default=0
)
parser.add_argument(
    "--opaque",
    type=int,
    default=0
)
parser.add_argument(
    "--rand_bk",
    type=int,
    default=0
)
parser.add_argument(
    "--kernel_type",
    type=str,
    default="rand_GD"
)
parser.add_argument(
    "--channel_wise_kernel",
    type=int,
    default=1
)
parser.add_argument(
    "--patch_argmin",
    type=int,
    default=1
)
parser.add_argument(
    "--unbounded",
    type=int,
    default=0
)
parser.add_argument(
    "--lr_tonemapping",
    type=float,
    default=0.05
)
parser.add_argument(
    "--lrate_decay",
    type=int,
    default=10
)
parser.add_argument(
    "--fine_res",
    type=int,
    default=4096
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

# dataset 관련 파라미터 조절해야됨

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.deblur3 import SubjectLoader

    # training parameters
    max_steps = 2000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004

else:
    from datasets.deblur3 import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

weight_decay = args.wdecay
near_plane = 0.0
far_plane = 1.0
_type = args.type
rand_bk = ""
if args.rand_bk:
    train_dataset_kwargs = {"color_bkgd_aug": "random"}
    rand_bk = "rand_bk"

pad = (args.ksize - 1) // 2
pad_patch_size = pad*2 + args.patch_size
    
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    num_patches= args.num_patches,
    patch_size=args.patch_size,
    pad=pad,
    # focusmap
    fmo=args.fmo,
    tag=args.dataset_tag,
    lof=args.lof,
    grouping=args.grouping,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

# aabb = torch.concat((train_dataset.aabb[0], train_dataset.aabb[1])).to(torch.float32)
estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2**10)
valid_lof = args.lof - args.noconv
radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1], max_resolution=args.fine_res, ksize=args.ksize, valid_lof=valid_lof, N_train_img=train_dataset.N, 
                                    channel_wise=args.channel_wise_kernel, kernel_type=args.kernel_type, gamma=args.gamma,
                                    focus_map=train_dataset.focus_map, focus_lv=train_dataset.focus_lv).to(device)
grad_vars = list(radiance_field.parameters()) # SH, Hash, HashMLP, MLP, kernel
optimizer = torch.optim.Adam(
    [
        {'params': grad_vars[1], 'lr': args.lr_ngp, 'eps':1e-15},    # Hash
        {'params': grad_vars[2], 'lr': args.lr_ngp, 'weight_decay':1e-6},  # Hash mlp
        {'params': grad_vars[3], 'lr': args.lr_ngp, 'weight_decay':1e-6},  # mlp
        {'params': grad_vars[4], 'lr': args.lr_kernel}, # kernel
    ]
)
print(grad_vars[1].shape, grad_vars[1].shape[0] / 16)

scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

if args.noconv != 0 and (args.lof <= args.noconv or args.noconv / args.lof <= 0.2):
    exit(1)

tag = f"{args.tag}_lof{args.lof}_off{args.noconv}_{args.grouping}_fineres_{args.fine_res}_lrk{args.lr_kernel}_lrn{args.lr_ngp}_kdecay{args.lr_kernel_decay_target_ratio}_wdecay{weight_decay}_g{args.gamma}_{rand_bk}"
print(tag)
logdir = f"./log_occ/{args.scene}_{tag}"
os.makedirs(logdir, exist_ok=True)

focus_lv = train_dataset.focus_lv
mid_psnr = []
mid_lpips = []
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(logdir + "/log.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(args)
print(args)
if args.lr_kernel_decay_target_ratio > 0:
    kernel_lr_factor = args.lr_kernel_decay_target_ratio**(1/max_steps)
final_train_psnr = 0

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]
    rays_info = data["rays_info"]
    img_list = data["img_list"]
    focus_lv = data["focus_lv"]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
    )

    # render
    rgb, acc, depth, n_rendering_samples = render_image_with_occgrid_deblur(
        radiance_field,
        estimator,
        rays,
        # rendering options
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
        rays_info=rays_info,
        pad=pad,
        patch_size=args.patch_size,
        pad_patch_size=pad_patch_size,
        num_patches=args.num_patches,
        ksize=args.ksize,
        noconv=args.noconv,
        valid_lof=valid_lof,
        focus_lv=focus_lv,
        img_list=img_list,
        gamma=args.gamma,
        tonemapping = None,
        gt=pixels,
        patchwise_argmin=args.patch_argmin
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    owow = torch.tensor(loss.detach().item())
    if owow.isnan().any():
        logger.info(f"{step} nan")
        with open("./res_occ.txt", "a") as f:
            f.write(f"occ_{args.scene}_{tag}: nan @ step {step}\n")
        print(f"occ_{args.scene}_{tag}: nan @ step {step}\n")
        if step < 5001:
            shutil.rmtree(logdir, ignore_errors=True)
            os.rmdir(logdir)
        exit(1)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 1000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )
        logger.info(f"iter {step:05d}: training psnr: {psnr:.2f}")
        final_train_psnr = psnr

    if step > 0 and (step % max_steps == 0 or  step % 5000 == 0):
        # evaluation
        radiance_field.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        training_time = time.time() - tic
        import gc
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                torch.cuda.empty_cache()
                gc.collect()
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _ = render_image_with_occgrid_test(
                    128,    # max samples?
                    # scene
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                    gamma=args.gamma
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                # if i == 0:
                #     imageio.imwrite(
                #         "rgb_test.png",
                #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                #     )
                #     imageio.imwrite(
                #         "rgb_error.png",
                #         (
                #             (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                #         ).astype(np.uint8),
                #     )
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        if step % max_steps == 0:
            print(f"occ_{args.scene}_{tag}_evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, tr.time={int(training_time)}sec")
            with open("./res_occ.txt", "a") as f:
                f.write(f"occ_{args.scene}_{tag}: psnr_avg={psnr_avg:.2f}\t train={final_train_psnr:.2f}\tlpips_avg={lpips_avg:.4f}\ttr.time={int(training_time)}sec\n")
            with open(f"./res_occ_{args.scene}.txt", "a") as f:
                f.write(f"occ_{args.scene}_{tag}: psnr_avg={psnr_avg:.2f}\t train={final_train_psnr:.2f}\tlpips_avg={lpips_avg:.4f}\ttr.time={int(training_time)}sec\n")
            logger.info(f"iter {step:05d}: final_test_psnr: {psnr:.2f}")
            with open("./res_occ_int.txt", "a") as f:
                for i, (p, l) in enumerate(zip(mid_psnr, mid_lpips)):
                    f.write(f"occ_{args.scene}_{tag} @ {(i+1)*5000} : psnr_avg={p:.2f}, lpips_avg={l:.4f}, tr.time={int(training_time)}sec\n")

            if args.render_train:
                train_dataset = SubjectLoader(
                    subject_id=args.scene,
                    root_fp=args.data_root,
                    split="train",
                    num_rays=None,
                    device=device,
                    **test_dataset_kwargs,
                )
                with torch.no_grad():
                    os.makedirs(f"{logdir}/train", exist_ok=True)
                    import warnings
                    warnings.filterwarnings("ignore")
                    for i in tqdm.tqdm(range(len(train_dataset))):
                        data = train_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]

                        # rendering
                        rgb, acc, depth, _, = render_image_with_propnet_deblur_real(
                            radiance_field,
                            proposal_networks,
                            estimator,
                            rays,
                            # rendering options
                            num_samples=num_samples,
                            num_samples_per_prop=num_samples_per_prop,
                            near_plane=near_plane,
                            far_plane=far_plane,
                            sampling_type=sampling_type,
                            opaque_bkgd=opaque_bkgd,
                            render_bkgd=render_bkgd,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                            gamma=args.gamma
                        )
                        mse = F.mse_loss(rgb, pixels)
                        imageio.imwrite(f"{logdir}/train/{i:03d}.png", rgb.cpu().numpy())
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        lpips.append(lpips_fn(rgb, pixels).item())
                print(f"training_test: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
                # with open("./res.txt", "a") as f:
                #     f.write(f"occ_{args.scene}_train: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}\n")

            if args.vis_kernel:
                import torch.nn.functional as F
                os.makedirs(f"{logdir}/kernel", exist_ok=True)
                kk = radiance_field.kernel[0]
                if kk.ndim == 4:    # noimgwise
                    kk = kk.unsqueeze(0)
                N = kk.shape[0]
                lof = kk.shape[1]
                C = radiance_field.kernel_C
                kk = kk.view(N,lof,C,-1)
                kk = F.softmax(kk, dim=-1)
                kk = kk.view(N,lof,C,args.ksize,args.ksize)
                kk = kk.permute(0,2,3,1,4).reshape(N,C,args.ksize,lof*args.ksize)
                ret = torch.zeros((C, N*args.ksize, lof*args.ksize)).to(device)    # 01 1314 2627
                owow = torch.arange(0, lof*args.ksize, args.ksize)
                import imageio                                                                                                                                                                                                                                                                                                                                       
                for i in range(N):
                    for c in range(C):
                        ret[c,i*args.ksize: (i+1)*args.ksize] = kk[i,c]
                ret = ret.detach().cpu().numpy()
                for i in range(C):
                    imageio.imwrite(f"{logdir}/kernel/{i}.png", ret[i])
        else:
            mid_psnr.append(psnr_avg)
            mid_lpips.append(lpips_avg)
            print(f"iter {step}: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, tr.time={int(training_time)}sec")
            logger.info(f"iter {step:05d}: test_psnr: {psnr:.2f}")
