"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
## Patchwise + deblurring
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
import shutil

from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    LLFF_SCENES,
    render_image_with_propnet_deblur_real,
    set_random_seed,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

img2mse = lambda x, y : torch.mean((x - y) ** 2)
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
set_random_seed(20211202)

print(args)
from datasets.deblur3 import SubjectLoader
# training parameters
synthetic = False
max_steps = args.iteration
init_batch_size = args.batch_size
weight_decay = args.wdecay

# scene parameters
# aabb = torch.tensor([-1.5, -1.67, -1.0, 1.5, 1.67, 1.0], device=device)     # aabb 바꾼 버전도 해보기. 0~1로 && 백그라운드 켜보기도 해야함
near_plane = 0.0
far_plane = 1.0
_type = args.type

if _type in ["mine", "360"]:
    train_dataset_kwargs = {"color_bkgd_aug": "random"}
    test_dataset_kwargs = {}
elif _type == "synthetic":
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
else:
    raise NotImplementedError

unbounded = bool(args.unbounded)


opaque = ""
rand_bk = ""
if args.opaque:
    opaque_bkgd = True
    opaque = "opaque"
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

# aabb = torch.tensor([-1.5, -1.67, -1.0, 1.5, 1.67, 1.0], device=device) 
aabb = torch.concat((train_dataset.aabb[0], train_dataset.aabb[1])).to(torch.float32)
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

# weight decay (1e-6) to NN, but not to hash table entries
# setup the radiance field we want to train.

lr_prop = args.lr_ngp if args.lr_prop < 0 else args.lr_prop

prop_optimizer = torch.optim.RAdam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=lr_prop,
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
valid_lof = args.lof - args.noconv
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded, max_resolution=args.fine_res, ksize=args.ksize, 
                                    valid_lof=valid_lof, N_train_img=train_dataset.N, channel_wise=args.channel_wise_kernel, kernel_type=args.kernel_type, gamma=args.gamma,
                                    focus_map=train_dataset.focus_map, focus_lv=train_dataset.focus_lv).to(device)

# params = list(radiance_field.parameters())
# grad_vars = []
# for i in range(len(params)-1):
#     if i < 3:
#         grad_vars.append({'params': params[i], 'lr': args.lr_ngp})
#     else:
#         grad_vars.append({'params': params[i], 'lr': args.lr_tonemapping})
# grad_vars.append({'params': params[-1], 'lr': args.lr_kernel})

# import pdb; pdb.set_trace() # 왜 2 ** 19 * 2 * 16이 등록된 파라미터수보다 더 크지?;
# optimizer = torch.optim.RAdam(
#     grad_vars,
#     eps=1e-15,
#     weight_decay=weight_decay,
# )
grad_vars = list(radiance_field.parameters()) # SH, Hash, HashMLP, MLP, kernel
optimizer = torch.optim.RAdam(
    [
        {'params': grad_vars[1], 'lr': args.lr_ngp, 'eps':1e-15},    # Hash
        {'params': grad_vars[2], 'lr': args.lr_ngp, 'weight_decay':1e-6},  # Hash mlp
        {'params': grad_vars[3], 'lr': args.lr_ngp, 'weight_decay':1e-6},  # mlp
        {'params': grad_vars[4], 'lr': args.lr_kernel}, # kernel
    ]
)
print(grad_vars[1].shape, grad_vars[1].shape[0] / 16)
# import pdb; pdb.set_trace()
# scheduler = torch.optim.lr_scheduler.ChainedScheduler(
#     [
#         torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=0.01, total_iters=100
#         ),
#         torch.optim.lr_scheduler.MultiStepLR(
#             optimizer,
#             milestones=[
#                 max_steps // 2,
#                 max_steps * 3 // 4,
#                 max_steps * 9 // 10,
#             ],
#             gamma=0.33,
#         ), 아레 scheduler.step도 주석 풀러여ㅑ함
#     ]
# )
proposal_requires_grad_fn = get_proposal_requires_grad_fn()
# proposal_annealing_fn = get_proposal_annealing_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

if args.noconv != 0 and (args.lof <= args.noconv or args.noconv / args.lof <= 0.2):
    exit(1)

tag = f"{args.tag}_lof{args.lof}_off{args.noconv}_{args.grouping}_fineres_{args.fine_res}_lrk{args.lr_kernel}_lrn{args.lr_ngp}_lrp{lr_prop}_kdecay{args.lr_kernel_decay_target_ratio}_wdecay{weight_decay}_g{args.gamma}_{opaque}_{rand_bk}"
print(tag)
logdir = f"./log/{args.scene}_{tag}"

os.makedirs(logdir, exist_ok=True)

# get focus map
if 'focus_lv' in dir(radiance_field):
    focus_lv = radiance_field.focus_lv
    valid_lof = radiance_field.valid_lof
    del radiance_field.focus_lv
    del radiance_field.valid_lof
    noconv = 1
else:
    focus_lv = train_dataset.focus_lv
    noconv = args.noconv
    
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
if args.lr_kernel_decay_target_ratio > 0:
    kernel_lr_factor = args.lr_kernel_decay_target_ratio**(1/max_steps)
final_train_psnr = 0
# training
tic = time.time()
for step in tqdm.tqdm(range(max_steps + 1)):
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
    img_list = data["img_list"]
    focus_lv = data["focus_lv"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras = render_image_with_propnet_deblur_real(
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
        rays_info=rays_info,
        pad=pad,
        patch_size=args.patch_size,
        pad_patch_size=pad_patch_size,
        num_patches=args.num_patches,
        ksize=args.ksize,
        noconv=noconv,
        valid_lof=valid_lof,
        focus_lv=focus_lv,
        img_list=img_list,
        gamma=args.gamma,
        tonemapping = None,
        gt=pixels,
        patchwise_argmin=args.patch_argmin
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    # loss = F.smooth_l1_loss(rgb, pixels)
    loss = img2mse(rgb, pixels)

    owow = torch.tensor(loss.detach().item())
    if owow.isnan().any():
        logger.info(f"{step} nan")
        with open("./res_deb.txt", "a") as f:
            f.write(f"prop_{args.scene}_{tag}: nan @ step {step}\n")
        print(f"prop_{args.scene}_{tag}: nan @ step {step}\n")
        if step < 5001:
            shutil.rmtree(logdir, ignore_errors=True)
            os.rmdir(logdir)
        exit(1)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()  # ???
    # loss.backward()
    optimizer.step()
    # scheduler.step()
    decay_rate = 0.1
    decay_steps = args.lrate_decay * 1000
    new_lrate = args.lr_ngp * (decay_rate ** (step / decay_steps))

    if args.lr_kernel_decay_target_ratio == -1: # 커널도 nerfacc 방식으로 decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
    else:
        for param_group in optimizer.param_groups[:3]:  # 커널 제외
            param_group['lr'] = new_lrate

    if args.lr_kernel_decay_target_ratio > 0:
        optimizer.param_groups[-1]["lr"] = optimizer.param_groups[-1]["lr"] * kernel_lr_factor

    if step % 1000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )
        logger.info(f"iter {step:05d}: training psnr: {psnr:.2f}")
        final_train_psnr = psnr

    if step > 0 and (step % max_steps == 0 or  step % 5000 == 0):
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
                if step % max_steps == 0:
                    imageio.imwrite(f"{logdir}/{i:03d}.png", (rgb.cpu().numpy()*255).astype(np.uint8))
                else:
                    os.makedirs(f"{logdir}/intermediate", exist_ok=True)
                    imageio.imwrite(f"{logdir}/intermediate/{step}_{i:03d}.png", (rgb.cpu().numpy()*255).astype(np.uint8))
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
        if step % max_steps == 0:
            print(f"prop_{args.scene}_{tag}_evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, tr.time={int(training_time)}sec")
            with open("./res_deb.txt", "a") as f:
                f.write(f"prop_{args.scene}_{tag}: psnr_avg={psnr_avg:.2f}\t train={final_train_psnr:.2f}\tlpips_avg={lpips_avg:.4f}\ttr.time={int(training_time)}sec\n")
            with open(f"./res_deb_{args.scene}.txt", "a") as f:
                f.write(f"prop_{args.scene}_{tag}: psnr_avg={psnr_avg:.2f}\t train={final_train_psnr:.2f}\tlpips_avg={lpips_avg:.4f}\ttr.time={int(training_time)}sec\n")
            logger.info(f"iter {step:05d}: final_test_psnr: {psnr:.2f}")
            with open("./res_deb_int.txt", "a") as f:
                for i, (p, l) in enumerate(zip(mid_psnr, mid_lpips)):
                    f.write(f"prop_{args.scene}_{tag} @ {(i+1)*5000} : psnr_avg={p:.2f}, lpips_avg={l:.4f}, tr.time={int(training_time)}sec\n")

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
                        imageio.imwrite(f"{logdir}/train/{i:03d}.png", (rgb.cpu().numpy()*255).astype(np.uint8))
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        lpips.append(lpips_fn(rgb, pixels).item())
                print(f"training_test: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
                with open("./res.txt", "a") as f:
                    f.write(f"prop_{args.scene}_train: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}\n")

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
                    imageio.imwrite(f"{logdir}/kernel/{i}.png", (ret[i]*255).astype(np.uint8))
        else:
            mid_psnr.append(psnr_avg)
            mid_lpips.append(lpips_avg)
            print(f"iter {step}: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, tr.time={int(training_time)}sec")
            logger.info(f"iter {step:05d}: test_psnr: {psnr:.2f}")


