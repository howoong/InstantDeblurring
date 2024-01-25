#!/bin/bash

LLFF=(fern flower fortress horns leaves orchids trex room)
REAL_D=(seal cake caps cisco coral cupcake cups daisy tools sausage)
REAL_M=(ball basket buick coffee decoration girl heron parterre puppet stair)
SYN=(cozy2room factory pool tanabata wine)

# CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene sausage --data_root ~/nfs/DCT/data/deblur/real_defocus_blur --type synthetic --tag synthetic_1e-6 --wdecay 1e-6
if [ $1 == 0 ]
then
    echo "."
    for wdecay in 1e-4 5e-4
    do
        for obj in ${REAL_D[@]}
        do
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/real_defocus_blur --type synthetic --tag baseline --wdecay $wdecay
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/real_defocus_blur --type synthetic --tag baseline_randbk --wdecay $wdecay --rand_bk 1
        done
    done


elif [ $1 == 1 ]
then
    for wdecay in 1e-4 5e-4
    do
        for obj in ${SYN[@]}
        do
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/synthetic_defocus_blur --type synthetic --tag defocus_baseline --wdecay $wdecay
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/synthetic_defocus_blur --type synthetic --tag defocus_baseline_randbk --wdecay $wdecay --rand_bk 1
        done
    done
    for wdecay in 1e-4 5e-4
    do
        for obj in ${SYN[@]}
        do
            # CUDA_VISIBLE_DEVICES=0 python3 examples/train_ngp_nerf_prop_deblur.py --scene factory --data_root ~/nfs/DCT/data/deblur/synthetic_camera_motion_blur --type synthetic --tag motion_baseline --wdecay 0
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/synthetic_camera_motion_blur --type synthetic --tag motion_baseline --wdecay $wdecay
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/synthetic_camera_motion_blur --type synthetic --tag motion_baseline_randbk --wdecay $wdecay --rand_bk 1
        done
    done
    
elif [ $1 == 2 ]
then
    echo "."
    for wdecay in 1e-4 5e-4
    do
        for obj in ${REAL_M[@]}
        do
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/real_camera_motion_blur --type synthetic --tag baseline --wdecay $wdecay
            CUDA_VISIBLE_DEVICES=$1 python3 examples/train_ngp_nerf_prop_deblur.py --scene $obj --data_root ~/nfs/DCT/data/deblur/real_camera_motion_blur --type synthetic --tag baseline_randbk --wdecay $wdecay --rand_bk 1
        done
    done

fi

# prop_tools_base_lof256_off100_quantile_lrk0.1_lrn0.01_kdecay0.5_wdecay0.0_g2.2
# CUDA_VISIBLE_DEVICES=2 python3 examples/train_ngp_nerf_prop_deblur3.py --scene tools --data_root ~/nfs/DCT/data/deblur/real_defocus_blur --type synthetic --wdecay 0 --lof 256 --noconv 100 --dataset_tag tmp --grouping quantile --lr_kernel 1e-1 --gamma=2.2 --lr_kernel_decay_target_ratio 0.5 --lr_ngp 1e-2 --tag opaque --render_train 1 --vis_kernel 1

# CUDA_VISIBLE_DEVICES=1 python3 examples/train_ngp_nerf_prop_deblur3.py --scene tools --data_root ~/nfs/DCT/data/deblur/real_defocus_blur --type synthetic --wdecay 0 --lof 400 --noconv 0 --dataset_tag tmp --grouping quantile --lr_kernel 1e-1 --gamma=2.2 --lr_kernel_decay_target_ratio 0.0 --lr_ngp 1e-2 --tag testt --render_train 1 --vis_kernel 1

# 일단은 베스트모델(syn?360? wdecay?)를 찾기. 
# 커널 관련 정의하고 옵티마이저 설정하고 연결만 다들 잘 해주면 얼추 끝나긴함

#  CUDA_VISIBLE_DEVICES=2 python3 examples/train_ngp_nerf_prop_deblur.py --scene tools --data_root ~/nfs/DCT/data/deblur/real_defocus_blur --type synthetic --tag test25600 --wdecay 0


# 다음은 랜덤 백그라운드
# 360처럼 프로포절을 쓰되, 유니폼하게 하면? 
    # 프로포절관련세팅, 오페이크, 유니폼/lindisp, bkcolor 이거 16가지 조합으로 다 테스트 해보기
    # 에폭도 늘려야하지않나?
    # 베스트에 대해서 에폭 늘려보기. 그리고 웨이트 디케이 같은 튜닝은 간단하게 Llff에서 해보고 디블러로 넘어가기. 근데 ㅅㅂ 1만에서 tr psnr이 27.25인데 다른 로스 안줘도 되나?;;
# 그 다음이 풀 360. 이거까지 다 하고나서 디블러 돌려보기
# 일단 신테틱, 360으로 모든 씬 돌려보고 신테틱에서도 디케이를 1e-6으로 바꿔서도 돌려보기
# CUDA_VISIBLE_DEVICES=0 python3 examples/train_ngp_nerf_prop.py --scene fern --data_root ~/nfs/DCT/data/nerf_llff_data --tag test --type 360


# 근데 지금 ray는 0~1로 해놓고 Lindsp 쓰는게 말이 되는건가? 360처럼 할거면 레이도 다시 해야되는 것 아님?
# 360데이터셋 기주느로 로드 콜맵에서 불러서 만든 최종 c2w, K와 포즈그거 로드한게 같으면 완전히 360 방식으로 할 수 있긴 함
# 모델 관련 - 360 유니폼? 즉 2개의 프로포절을 쓰되, 유니폼하게 나누는 것. 아니 근데 360이 룸이 이상하게 안나옴. 트레인은 37.18까지 가는데