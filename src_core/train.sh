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
    --partial_leak_scale 0.2 0.7

# PSF mode example:
# python trainer.py \
#     --bs 16 --lr 0.001 --epochs 30 --gpu_id 0 \
#     --checkpoint_path ../checkpoints/4channel \
#     --patch_size 128 \
#     --training_dataset_path ../data/grid_stripe_4channel/train/good/ \
#     --img_format tiff --num_defects_range 4 10 --cache_size 100 \
#     --defect_mode psf --psf_type type4_vector \
#     --partial_leak_scale 0.2 0.7

# Notes:
#   --partial_leak_scale MIN MAX
#       Some target-only defects also appear at reduced intensity in one ref
#       (simulating refs partially seeing the defect). Probability is fixed
#       at 0.4 per target-only defect; this flag controls the leak intensity
#       range as a fraction of the full defect intensity. Set 0 0 to disable
#       (the leak still picks a scale of 0 if it triggers, so refs are
#       effectively unchanged).
