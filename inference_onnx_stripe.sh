python inference_onnx.py \
    --model_path ./onnx_models/background_removal_stripe.onnx \
    --test_path ./MVTec_AD_dataset/grid_stripe_4channel/test/ \
    --output_dir ./output/onnx_stripe \
    --img_format tiff \
    --image_type strip
