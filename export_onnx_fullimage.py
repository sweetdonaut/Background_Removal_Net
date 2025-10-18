import torch
import argparse
import os
import numpy as np
from model import SegmentationNetwork, SegmentationNetworkONNX, SegmentationNetworkONNXFullImage


def export_fullimage_to_onnx(checkpoint_path, output_path, opset_version=11):
    """
    Export trained model to ONNX format for FULL IMAGE processing

    This version includes sliding window logic inside the ONNX model.
    - Input: (1, 3, 976, 176) - Full strip image
    - Output: (1, 3, 976, 176) - Full heatmap

    Args:
        checkpoint_path: Path to trained model checkpoint (.pth)
        output_path: Path to save ONNX model (.onnx)
        opset_version: ONNX opset version (default: 11)
    """
    print("=" * 80)
    print("ONNX Full Image Export Script")
    print("=" * 80)

    # Load checkpoint
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model configuration
    patch_height = checkpoint['img_height']
    patch_width = checkpoint['img_width']
    image_type = checkpoint.get('image_type', 'unknown')

    print(f"   - Patch size: {patch_height}x{patch_width}")
    print(f"   - Image type: {image_type}")
    print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")

    if image_type != 'strip':
        print(f"\n⚠️  Warning: This export script is designed for 'strip' images.")
        print(f"   Current image type: {image_type}")
        print(f"   Proceeding anyway, but output may not be correct.")

    # Create base model
    print("\n2. Creating base segmentation network...")
    base_model = SegmentationNetwork(in_channels=3, out_channels=2)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # Wrap with patch-level ONNX wrapper (adds softmax + 3 channels)
    print("3. Creating patch-level ONNX wrapper...")
    patch_onnx_model = SegmentationNetworkONNX(base_model)
    patch_onnx_model.eval()

    # Wrap with full-image ONNX wrapper (adds sliding window)
    print("4. Creating full-image ONNX wrapper (with sliding window)...")
    full_image_model = SegmentationNetworkONNXFullImage(patch_onnx_model)
    full_image_model.eval()

    # Create dummy input for FULL IMAGE
    print(f"\n5. Creating dummy input: (1, 3, 976, 176)")
    dummy_input = torch.randn(1, 3, 976, 176)

    # Test forward pass
    print("6. Testing forward pass...")
    with torch.no_grad():
        output = full_image_model(dummy_input)
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"   - Channel 0 (anomaly) range: [{output[0, 0].min():.4f}, {output[0, 0].max():.4f}]")
        print(f"   - Channel 1 (zero) sum: {output[0, 1].sum():.6f}")
        print(f"   - Channel 2 (zero) sum: {output[0, 2].sum():.6f}")

    # Export to ONNX
    print(f"\n7. Exporting to ONNX (opset version {opset_version})...")
    print("   ⏳ This may take a moment (embedding sliding window logic)...")

    torch.onnx.export(
        full_image_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"   ✓ ONNX model saved to: {output_path}")

    # Verify ONNX model
    print("\n8. Verifying ONNX model...")
    try:
        import onnx
        onnx_model_check = onnx.load(output_path)
        onnx.checker.check_model(onnx_model_check)
        print("   ✓ ONNX model is valid!")

        # Print model info
        print("\n9. ONNX Model Information:")
        print(f"   - IR version: {onnx_model_check.ir_version}")
        print(f"   - Producer: {onnx_model_check.producer_name}")
        print(f"   - Graph inputs: {[i.name for i in onnx_model_check.graph.input]}")
        print(f"   - Graph outputs: {[o.name for o in onnx_model_check.graph.output]}")

    except ImportError:
        print("   ⚠ ONNX package not found. Skipping verification.")

    # Test with ONNX Runtime
    print("\n10. Testing with ONNX Runtime...")
    try:
        import onnxruntime as ort

        # Create session
        ort_session = ort.InferenceSession(output_path)

        # Prepare input
        dummy_input_np = dummy_input.numpy()

        # Run inference
        ort_outputs = ort_session.run(None, {'input': dummy_input_np})
        ort_output = ort_outputs[0]

        # Compare with PyTorch
        pytorch_output = output.numpy()
        diff = np.abs(pytorch_output - ort_output)

        print(f"   - ONNX Runtime output shape: {ort_output.shape}")
        print(f"   - Max difference (PyTorch vs ONNX): {diff.max():.6e}")

        if diff.max() < 1e-4:
            print("   ✓ ONNX Runtime output matches PyTorch!")
        else:
            print(f"   ⚠ Warning: Difference detected (max: {diff.max():.6e})")

        # Verify channels
        print(f"\n   Channel verification:")
        print(f"   - Channel 0 (anomaly) has values: {ort_output[0, 0].sum() > 0}")
        print(f"   - Channel 1 sum (should be 0): {ort_output[0, 1].sum():.6f}")
        print(f"   - Channel 2 sum (should be 0): {ort_output[0, 2].sum():.6f}")

    except ImportError:
        print("   ⚠ ONNX Runtime not found. Skipping runtime test.")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 80)
    print("Export Summary")
    print("=" * 80)
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ ONNX Model: {output_path}")
    print(f"✓ File Size: {file_size_mb:.2f} MB")
    print(f"")
    print(f"✓ Input Shape: (1, 3, 976, 176) - FULL IMAGE")
    print(f"✓ Output Shape: (1, 3, 976, 176) - FULL HEATMAP")
    print(f"")
    print(f"✓ Sliding Window: EMBEDDED IN ONNX (9 Y × 2 X patches)")
    print(f"✓ Output Channel 0: Anomaly heatmap (probability)")
    print(f"✓ Output Channel 1-2: Zero-filled placeholders")
    print("=" * 80)
    print("\n🎯 Ready for production deployment!")
    print("   This ONNX model processes FULL images in a single pass.")
    print("   No external preprocessing needed - just feed the image!")
    print()


def main():
    parser = argparse.ArgumentParser(description='Export trained model to ONNX (Full Image version)')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save ONNX model (.onnx file)')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX opset version (default: 11)')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Export
    export_fullimage_to_onnx(args.checkpoint_path, args.output_path, args.opset_version)


if __name__ == "__main__":
    main()
