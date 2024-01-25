import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os

def compute_blur_measure(image, patch_size):
    # Calculate the number of overlapping patches in both dimensions
    image = torch.from_numpy(image)
    pad = (patch_size - 2) // 2
    image = image.unsqueeze(0).float()
    image = F.pad(image, (pad+1,pad,pad+1,pad), mode="reflect").numpy()
    image = image.squeeze()
    height, width = image.shape
    print(image.shape)
    # stride = patch_size // 2
    # if stride == 0:
    #     stride = 1
    stride = 1
    num_patches_vert = (height - patch_size) // stride + 1
    num_patches_horiz = (width - patch_size) // stride + 1

    # Initialize an array to store the blur measure for each patch
    blur_measures = np.zeros((num_patches_vert, num_patches_horiz))

    # Process each patch and compute the blur measure
    for i in range(num_patches_vert):
        for j in range(num_patches_horiz):
            y_start = i * stride
            x_start = j * stride
            patch = image[y_start:y_start + patch_size, x_start:x_start + patch_size]

            # Compute the Fourier Transform
            fft_patch = np.fft.fft2(patch)

            # Shift zero frequency components to the center
            fft_patch_shifted = np.fft.fftshift(fft_patch)

            # Compute the magnitude spectrum (log-scaled for visualization)
            magnitude_spectrum = np.log(np.abs(fft_patch_shifted) + 1)

            # Calculate the blur measure using the sum of energy in low frequencies
            blur_measures[i, j] = np.sum(magnitude_spectrum)

    return blur_measures

def vis(patch_size, image_path, obj, img_num):
    from PIL import Image
    if patch_size > 0:
        focus_map = np.load(f"patch_{obj}_{patch_size}_img{img_num}.npy")
    else:
        focus_map = np.load("/home/blee/nfs/DCT/data/deblur/real_camera_motion_blur/blurcoffee/focus_map_SML_7.npy")[9]
    image = np.array(Image.open(image_path))

    _min = focus_map.min()
    _max = focus_map.max()
    _rng = _max - _min
    focus_map = (focus_map - _min) / _rng
    focus_map = 1 - focus_map
    import torch
    focus_map = torch.from_numpy(focus_map.flatten())
    val, idx = focus_map.sort()
    image = image.reshape(400*600, 3)
    split = 400 * 600 // 50

    import imageio
    

    # 패치 단위로 하되 스트라이드를 1로 잡고 양쪽에 패딩 붙여주면? SML도 마찬가지 아닌가?
    if patch_size > 0:
        os.makedirs(f"vis/{obj}_p{patch_size}_img{img_num}", exist_ok=True)
    else:
        os.makedirs(f"vis/{obj}_SML_img{img_num}", exist_ok=True)

    for i in range(50):
        ret = np.zeros_like(image)
        tar = idx[i*split : (i+1)*split]
        ret[tar] = image[tar]
        if patch_size > 0:
            imageio.imwrite(f"vis/{obj}_p{patch_size}_img{img_num}/{i}.png", ret.reshape(400,600, 3))
        else:
            imageio.imwrite(f"vis/{obj}_SML_img{img_num}/{i}.png", ret.reshape(400,600, 3))

if __name__ == '__main__':

    for patch_size in [8,16,64]:
        # for obj in ["ball", "basket", "buick", "decoration", "girl", "heron", "parterre", "puppet", "stair"]:
        for obj in ["heron", "parterre", "puppet", "stair"]:
            blurs = []
            li = f'/home/blee/nfs/DCT/data/deblur/real_camera_motion_blur/blur{obj}/images_4/'
            N = len(os.listdir(li))
            print(obj, N)
            for img_num in range(N):
                print(obj, img_num, N)
                image_path = f'/home/blee/nfs/DCT/data/deblur/real_camera_motion_blur/blur{obj}/images_4/{img_num:03d}.png'
                # patch_size = 32  # Adjust the patch size as desired (e.g., 8, 16, 32)


                if patch_size == -1:
                    vis(patch_size, image_path, obj, img_num)
                    exit(1)


                # Load the image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Compute the blur measure for each patch
                blur_measures = compute_blur_measure(image, patch_size)
                blurs.append(blur_measures)
                # np.save(f"patch_{obj}_{patch_size}_img{img_num}.npy", blur_measures)
                # print(blur_measures.shape)
                # vis(patch_size, image_path, obj, img_num)
            blurs = np.stack(blurs)
            print(blurs.shape)
            np.save(f'/home/blee/nfs/DCT/data/deblur/real_camera_motion_blur/blur{obj}/focus_map_fourier_{patch_size}.npy', blurs)
    exit(1)

    # Display the original image and the blur measure heatmap
    # plt.figure(figsize=(10, 5))

    plt.imshow(blur_measures, cmap='jet', interpolation='nearest')
    plt.title('Blur Measure Heatmap')
    plt.axis('off')
    plt.savefig(f"owow_blur{patch_size}.png")

    plt.show()
