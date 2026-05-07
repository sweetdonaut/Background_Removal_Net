python trainer.py \
    --bs 16 \
    --lr 0.001 \
    --epochs 30 \
    --gpu_id 0 \
    --checkpoint_path ../checkpoints/4channel \
    --patch_size 128 \
    --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
    --img_format tiff \
    --num_defects_range 4 10 \
    --cache_size 100 \
    --defect_mode gaussian \
    --partial_leak_prob 0.0 \
    --partial_leak_scale 0 0

# PSF mode example:
# python trainer.py \
#     --bs 16 --lr 0.001 --epochs 30 --gpu_id 0 \
#     --checkpoint_path ../checkpoints/4channel \
#     --patch_size 128 \
#     --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
#     --img_format tiff --num_defects_range 4 10 --cache_size 100 \
#     --defect_mode psf --psf_type type4_vector \
#     --partial_leak_prob 0.0 \
#     --partial_leak_scale 0 0

# Notes:
#   --partial_leak_prob P, --partial_leak_scale MIN MAX
#       Some target-only defects also appear at reduced intensity in one ref
#       (simulating refs partially seeing the defect). --partial_leak_prob is
#       the per-defect trigger chance; --partial_leak_scale is the intensity
#       fraction range when it does trigger. Both default to 0 (disabled).
