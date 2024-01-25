def batch_collate(batch):
    transposed = zip(*batch)
    it = iter(transposed)
    rays_train = next(it)
    rays_train = torch.concat(rays_train)
    rgbs_train = next(it)
    rgbs_train = torch.stack(rgbs_train)
    chunk = next(it)
    chunk = sum(chunk)
    img_list = next(it)
    img_list = list(img_list)
    ray_info = next(it)
    ray_info = list(ray_info)
    focus_lv_train = next(it)
    focus_lv_train = torch.stack(focus_lv_train)

    return rays_train, rgbs_train, chunk, img_list, ray_info, focus_lv_train

dataloader = DataLoader(train_dataset, batch_size=args.patch_batch, num_workers=0, shuffle=True, collate_fn=batch_collate, drop_last=True)
batch_iterator = iter(dataloader)

batch_iterator = iter(dataloader)


for iteration:
    try:
        rays_train, rgb_train, chunk, img_list, rays_info, focus_lv_train = next(batch_iterator)
    except StopIteration:
        batch_iterator = iter(dataloader)
        rays_train, rgb_train, chunk, img_list, rays_info, focus_lv_train = next(batch_iterator)
    rays_train = rays_train.to(device)
    rgb_train = rgb_train.view(-1,3).to(device)
    focus_lv_train = focus_lv_train.to(device)

    rgb_map, depth_map, sharp_rgb_map = renderer(rays_train, tensorf_coarse, blurring=blurring, img_list=img_list,            
            focus_lv=focus_lv_train, chunk=chunk, N_samples_coarse=nSamples_coarse, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True, pad_mode=pad_mode, rays_info=rays_info, top_off=args.top_off, gt=rgb_train)

    loss_coarse = torch.mean((rgb_map - rgb_train) ** 2)

