def foo(key, pad, patch_size):
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

def forward(self, rays_chunk, blurring=False, img_list=None, focus_lv=None, white_bg=True, is_train=False, ndc_ray=False, N_samples_coarse=-1, 
            rays_info=None, pad_mode="full", top_off=False, gt=None):

    if blurring:
        rgb_maps = []
        pos = 0
        for region, size, wh, pad_info in rays_info:
            cur_rgb_map = rgb_map[pos: pos+size]
            if region != "center9":
                if wh == None:
                    h, w, padding = foo(region, self.pad, self.patch_size)
                else:
                    w, h = wh
                    padding = (0,0) + pad_info
                cur_rgb_map = F.pad(cur_rgb_map.view(-1, h, w, 3), padding, 'reflect')
            else:
                cur_rgb_map = cur_rgb_map.view(-1, self.pad*2 + self.patch_size, self.pad*2 + self.patch_size, 3)
            rgb_maps.append(cur_rgb_map)
            pos += size
        pad_rgb_map = torch.concat(rgb_maps)
            
        sharp_rgb_map = pad_rgb_map[:,self.pad:-self.pad, self.pad:-self.pad].clamp(0,1)

        if self.kernel_N > 1:
            cur_kernel = self.kernel[0][img_list]  # N_patch, lv, ksize, ksize, C
        else:
            cur_kernel = self.kernel[0][None,...].repeat(self.patch_batch, 1, 1, 1, 1)

        # apply softmax to kernel to make its sum is equal to 1
        cur_kernel = cur_kernel.view(self.patch_batch,-1,self.ksize*self.ksize,self.kernel_C)    # N_patch, valid_lof, ksize**2, C
        cur_kernel = F.softmax(cur_kernel, dim=-2)
        cur_kernel = cur_kernel.view(self.patch_batch,-1,self.ksize,self.ksize,self.kernel_C)    # N_patch, valid_lof, ksize, ksize, C
        if self.kernel_C == 1:
            cur_kernel = cur_kernel.repeat(1,1,1,1,3)   # N_patch, valid_lof, ksize, ksize, C
        else:
            cur_rgb_map = pad_rgb_map.permute(0,3,1,2).reshape(1, -1, self.pad_patch_size, self.pad_patch_size) # 1, NC, P, P
        
        cur_kernel = torch.concat([cur_kernel, self.identity], dim=1)
        _focus_lv = focus_lv.view(self.patch_batch,-1) # IMG, FL
        _focus_lv = _focus_lv[...,None,None,None].repeat(1,1,self.ksize,self.ksize,3)
        blur_weight = torch.gather(cur_kernel, dim=1, index=_focus_lv)
        window_size = self.patch_size
        stride = 1
        cropped_img = pad_rgb_map.unfold(1, window_size, stride).unfold(2, window_size, stride).permute(0,4,5,1,2,3).reshape(self.patch_batch,-1,self.ksize,self.ksize,3)
        rgb_map = (blur_weight * cropped_img).sum(dim=(2,3)).view(-1,3)
                
    # Tonemapping
    if self.gamma > 0:
        rgb_map = rgb_map ** (1. / self.gamma)
        sharp_rgb_map = sharp_rgb_map ** (1. / self.gamma)
    elif self.gamma == -1:
        rgb_map = self.tonemapping(rgb_map)
    rgb_map = rgb_map.clamp(0,1)


    return rgb_map, depth_map, sharp_rgb_map