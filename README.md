# Instant Deblurring

Our code is based on NerfAcc (https://github.com/nerfstudio-project/nerfacc).

## Results

![000](https://github.com/howoong/InstantDeblurring/assets/68628830/307e233c-4bd6-48c2-8f7d-08c0c1e5e378)
![001](https://github.com/howoong/InstantDeblurring/assets/68628830/71d606ea-c7d8-4e33-983d-88843ef260b4)
![002](https://github.com/howoong/InstantDeblurring/assets/68628830/1fad1787-854f-4c77-82bf-bc091a19c72e)
![003](https://github.com/howoong/InstantDeblurring/assets/68628830/38259743-b510-4061-8e52-62921841f0d7)
![004](https://github.com/howoong/InstantDeblurring/assets/68628830/06e4606e-bdf2-4c3e-aa55-7061f9621b6e)
![005](https://github.com/howoong/InstantDeblurring/assets/68628830/bdb8db2f-4a68-4e04-8e61-195ca1b7c38f)
![006](https://github.com/howoong/InstantDeblurring/assets/68628830/7e87d897-ccdf-4e55-9c8f-3686f78467f4)

# Nerfacc

[News] 2023/04/04. If you were using `nerfacc <= 0.3.5` and would like to migrate to our latest version (`nerfacc >= 0.5.0`), Please check the [CHANGELOG](CHANGELOG.md) on how to migrate.

NerfAcc is a PyTorch Nerf acceleration toolbox for both training and inference. It focus on
efficient sampling in the volumetric rendering pipeline of radiance fields, which is 
universal and plug-and-play for most of the NeRFs.
With minimal modifications to the existing codebases, Nerfacc provides significant speedups 
in training various recent NeRF papers.
**And it is pure Python interface with flexible APIs!**

![Teaser](/docs/source/_static/images/teaser.jpg?raw=true)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easist way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).
```
pip install nerfacc
```

Or install from source. In this way it will build the CUDA code during installation.
```
pip install git+https://github.com/KAIR-BAIR/nerfacc.git
```

We also provide pre-built wheels covering major combinations of Pytorch + CUDA supported by [official Pytorch](https://pytorch.org/get-started/previous-versions/).

```
# e.g., torch 1.13.0 + cu117
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu117.html
```

| Windows & Linux | `cu113` | `cu115` | `cu116` | `cu117` | `cu118` |
|-----------------|---------|---------|---------|---------|---------|
| torch 1.11.0    | ✅      | ✅      |         |         |         |
| torch 1.12.0    | ✅      |         | ✅      |         |         |
| torch 1.13.0    |         |         | ✅      | ✅      |         |
| torch 2.0.0     |         |         |         | ✅      | ✅      |

For previous version of nerfacc, please check [here](https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/index.html) on the supported pre-built wheels.

## Usage

The idea of NerfAcc is to perform efficient volumetric sampling with a computationally cheap estimator to discover surfaces.
So NerfAcc can work with any user-defined radiance field. To plug the NerfAcc rendering pipeline into your code and enjoy 
the acceleration, you only need to define two functions with your radience field.

- `sigma_fn`: Compute density at each sample. It will be used by the estimator
  (e.g., `nerfacc.OccGridEstimator`, `nerfacc.PropNetEstimator`) to discover surfaces. 
- `rgb_sigma_fn`: Compute color and density at each sample. It will be used by 
  `nerfacc.rendering` to conduct differentiable volumetric rendering. This function 
  will receive gradients to update your radiance field.

An simple example is like this:

``` python
import torch
from torch import Tensor
import nerfacc 

radiance_field = ...  # network: a NeRF model
rays_o: Tensor = ...  # ray origins. (n_rays, 3)
rays_d: Tensor = ...  # ray normalized directions. (n_rays, 3)
optimizer = ...       # optimizer

estimator = nerfacc.OccGridEstimator(...)

def sigma_fn(
    t_starts: Tensor, t_ends:Tensor, ray_indices: Tensor
) -> Tensor:
    """ Define how to query density for the estimator."""
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)
    positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
    sigmas = radiance_field.query_density(positions) 
    return sigmas  # (n_samples,)

def rgb_sigma_fn(
    t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor
) -> Tuple[Tensor, Tensor]:
    """ Query rgb and density values from a user-defined radiance field. """
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)
    positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
    rgbs, sigmas = radiance_field(positions, condition=t_dirs)  
    return rgbs, sigmas  # (n_samples, 3), (n_samples,)

# Efficient Raymarching:
# ray_indices: (n_samples,). t_starts: (n_samples,). t_ends: (n_samples,).
ray_indices, t_starts, t_ends = estimator.sampling(
    rays_o, rays_d, sigma_fn=sigma_fn, near_plane=0.2, far_plane=1.0, early_stop_eps=1e-4, alpha_thre=1e-2, 
)

# Differentiable Volumetric Rendering.
# colors: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
color, opacity, depth, extras = nerfacc.rendering(
    t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
)

# Optimize: Both the network and rays will receive gradients
optimizer.zero_grad()
loss = F.mse_loss(color, color_gt)
loss.backward()
optimizer.step()
```

## Examples: 

Before running those example scripts, please check the script about which dataset is needed, and download the dataset first. You could use `--data_root` to specify the path.

```bash
# clone the repo with submodules.
git clone --recursive git://github.com/KAIR-BAIR/nerfacc/
```

### Static NeRFs

See full benchmarking here: https://www.nerfacc.com/en/stable/examples/static.html

Instant-NGP on NeRF-Synthetic dataset with better performance in 4.5 minutes.
``` bash
# Occupancy Grid Estimator
python examples/train_ngp_nerf_occ.py --scene lego --data_root data/nerf_synthetic
# Proposal Net Estimator
python examples/train_ngp_nerf_prop.py --scene lego --data_root data/nerf_synthetic
```

Instant-NGP on Mip-NeRF 360 dataset with better performance in 5 minutes.
``` bash
# Occupancy Grid Estimator
python examples/train_ngp_nerf_occ.py --scene garden --data_root data/360_v2
# Proposal Net Estimator
python examples/train_ngp_nerf_prop.py --scene garden --data_root data/360_v2
```

Vanilla MLP NeRF on NeRF-Synthetic dataset in an hour.
``` bash
# Occupancy Grid Estimator
python examples/train_mlp_nerf.py --scene lego --data_root data/nerf_synthetic
```

TensoRF on Tanks&Temple and NeRF-Synthetic datasets (plugin in the official codebase).
``` bash
cd benchmarks/tensorf/
# (set up the environment for that repo)
bash script.sh nerfsyn-nerfacc-occgrid 0
bash script.sh tt-nerfacc-occgrid 0
```

### Dynamic NeRFs

See full benchmarking here: https://www.nerfacc.com/en/stable/examples/dynamic.html

T-NeRF on D-NeRF dataset in an hour.
``` bash
# Occupancy Grid Estimator
python examples/train_mlp_tnerf.py --scene lego --data_root data/dnerf
```

K-Planes on D-NeRF dataset (plugin in the official codebase).
```bash
cd benchmarks/kplanes/
# (set up the environment for that repo)
bash script.sh dnerf-nerfacc-occgrid 0
```

TiNeuVox on HyperNeRF and D-NeRF datasets (plugin in the official codebase).
```bash
cd benchmarks/tineuvox/
# (set up the environment for that repo)
bash script.sh dnerf-nerfacc-occgrid 0
bash script.sh hypernerf-nerfacc-occgrid 0
bash script.sh hypernerf-nerfacc-propnet 0
```

### Camera Optimization NeRFs

See full benchmarking here: https://www.nerfacc.com/en/stable/examples/camera.html

BARF on the NeRF-Synthetic dataset (plugin in the official codebase).
```bash
cd benchmarks/barf/
# (set up the environment for that repo)
bash script.sh nerfsyn-nerfacc-occgrid 0
```

### 3rd-Party Usages:

#### Awesome Codebases.
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio): A collaboration friendly studio for NeRFs.
- [sdfstudio](https://autonomousvision.github.io/sdfstudio/): A unified framework for surface reconstruction.
- [threestudio](https://github.com/threestudio-project/threestudio): A unified framework for 3D content creation.
- [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl): NeuS in 10 minutes.
- [modelscope](https://github.com/modelscope/modelscope/blob/master/modelscope/models/cv/nerf_recon_acc/network/nerf.py): A collection of deep-learning algorithms.

#### Awesome Papers.
- [Representing Volumetric Videos as Dynamic MLP Maps, CVPR 2023](https://github.com/zju3dv/mlp_maps)
- [NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads, ArXiv 2023](https://tobias-kirschstein.github.io/nersemble/)
- [HumanRF: High-Fidelity Neural Radiance Fields for Humans in Motion, ArXiv 2023](https://synthesiaresearch.github.io/humanrf/)

## Common Installation Issues

<details>
    <summary>ImportError: .../csrc.so: undefined symbol</summary>
    If you are installing a pre-built wheel, make sure the Pytorch and CUDA version matchs with the nerfacc version (nerfacc.__version__).
</details>

