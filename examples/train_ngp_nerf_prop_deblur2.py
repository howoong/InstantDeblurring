"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
## Patchwise만 적용. 디블러잉까지는 X
import argparse
import itertools
import pathlib
import time
from typing import Callable

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPDensityField, NGPRadianceField

import os

from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    LLFF_SCENES,
    render_image_with_propnet_deblur,
    set_random_seed,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

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
    "--test_chunk_size",
    type=int,
    default=8192,
)
parser.add_argument(
    "--tag",
    type=str,
    default="test",
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
    "--rand_bk",
    type=int,
    default=0
)

args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

print(f"{args.scene}, {args.tag}, {args.type}!!!!!!!!")
from datasets.deblur2 import SubjectLoader
# training parameters
synthetic = False
max_steps = 20000
init_batch_size = 4096
weight_decay = args.wdecay

# scene parameters
aabb = torch.tensor([-1.5, -1.67, -1.0, 1.5, 1.67, 1.0], device=device)     # aabb 바꾼 버전도 해보기. 0~1로 && 백그라운드 켜보기도 해야함
near_plane = 0.0
far_plane = 1.0
_type = args.type

if _type in ["mine", "360"]:
    train_dataset_kwargs = {"color_bkgd_aug": "random"}
    test_dataset_kwargs = {}
    unbounded = True
elif _type == "synthetic":
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    unbounded = False
else:
    raise NotImplementedError

if _type in ["mine", "synthetic"]:
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    sampling_type = "uniform"
    opaque_bkgd = False

elif _type in ["360"]:
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=256,
        ).to(device),
    ]
    # render parameters
    num_samples = 48
    num_samples_per_prop = [256, 96]
    sampling_type = "lindisp"
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2  # TODO: Try 0.02
    far_plane = 1e3
    opaque_bkgd = True

if args.rand_bk == 1:
    train_dataset_kwargs = {"color_bkgd_aug": "random"}    
    
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
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

# setup the radiance field we want to train.
prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            prop_optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            prop_optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
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
proposal_requires_grad_fn = get_proposal_requires_grad_fn()
# proposal_annealing_fn = get_proposal_annealing_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

logdir = f"./log_baseline/{args.scene}_{args.tag}"
os.makedirs(logdir, exist_ok=True)
# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    for p in proposal_networks:
        p.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]
    rays_info = data["rays_info"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras = render_image_with_propnet_deblur(
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
        # train options
        proposal_requires_grad=proposal_requires_grad,
        rays_info=rays_info
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 10000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        for p in proposal_networks:
            p.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        training_time = time.time() - tic
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _, = render_image_with_propnet_deblur(
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
                )
                mse = F.mse_loss(rgb, pixels)
                imageio.imwrite(f"{logdir}/{i:03d}.png", rgb.cpu().numpy())
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
                #     break
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"prop_{args.scene}_{args.tag}_evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, tr.time={int(training_time)}sec")
        with open("./res_deb.txt", "a") as f:
            f.write(f"prop_{args.scene}_{args.tag}: psnr_avg={psnr_avg:.2f}, lpips_avg={lpips_avg:.4f}, tr.time={int(training_time)}sec\n")


        # train_dataset = SubjectLoader(
        #     subject_id=args.scene,
        #     root_fp=args.data_root,
        #     split="train",
        #     num_rays=None,
        #     device=device,
        #     **test_dataset_kwargs,
        # )
        # with torch.no_grad():
        #     os.makedirs(f"{logdir}/train", exist_ok=True)
        #     for i in tqdm.tqdm(range(len(train_dataset))):
        #         data = test_dataset[i]
        #         render_bkgd = data["color_bkgd"]
        #         rays = data["rays"]
        #         pixels = data["pixels"]

        #         # rendering
        #         rgb, acc, depth, _, = render_image_with_propnet(
        #             radiance_field,
        #             proposal_networks,
        #             estimator,
        #             rays,
        #             # rendering options
        #             num_samples=num_samples,
        #             num_samples_per_prop=num_samples_per_prop,
        #             near_plane=near_plane,
        #             far_plane=far_plane,
        #             sampling_type=sampling_type,
        #             opaque_bkgd=opaque_bkgd,
        #             render_bkgd=render_bkgd,
        #             # test options
        #             test_chunk_size=args.test_chunk_size,
        #         )
        #         mse = F.mse_loss(rgb, pixels)
        #         imageio.imwrite(f"{logdir}/train/{i:03d}.png", rgb.cpu().numpy())
        #         psnr = -10.0 * torch.log(mse) / np.log(10.0)
        #         psnrs.append(psnr.item())
        #         lpips.append(lpips_fn(rgb, pixels).item())
        # print(f"training_test: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
        # with open("./res.txt", "a") as f:
        #     f.write(f"prop_{args.scene}_train: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}\n")