python inference_onnx.py \
    --model_path ./onnx_models/background_removal_square.onnx \
    --test_path ./MVTec_AD_dataset/grid_align_3channel/test/ \
    --output_dir ./output/onnx_square \
    --img_format tiff \
    --image_type square
