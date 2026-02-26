import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
import glob as glob_module
import tifffile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_test_images(test_path, img_format='png_jpg', image_type='square'):
    """Load test images from directory"""
    image_paths = []

    if img_format == 'png_jpg':
        # Check main directory
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*.png")))
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*.jpg")))
        # Check subdirectories
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*", "*.png")))
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*", "*.jpg")))
    else:  # tiff
        # Check main directory
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*.tiff")))
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*.tif")))
        # Check subdirectories
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*", "*.tiff")))
        image_paths.extend(glob_module.glob(os.path.join(test_path, "*", "*.tif")))

    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} test images")

    return image_paths


def fullimage_inference(image, ort_session, image_type='strip'):
    """
    Perform full image inference using ONNX model

    This function directly processes the full image without manual sliding window.
    The ONNX model contains embedded sliding window logic.

    Args:
        image: Input image (H, W, 3) in uint8 or float32
        ort_session: ONNX Runtime inference session
        image_type: Type of images ('strip', 'square')

    Returns:
        output_heatmap: Anomaly heatmap (H, W) in range [0, 1]
        image: Original image
    """
    h, w = image.shape[:2]

    # Prepare input: (1, 3, H, W) in float32, normalized to [0, 1]
    three_channel = np.stack([image[:,:,0], image[:,:,1], image[:,:,2]], axis=0)
    three_channel_tensor = (three_channel / 255.0).astype(np.float32)
    three_channel_tensor = np.expand_dims(three_channel_tensor, 0)  # Add batch dimension

    # Get ONNX model input name
    input_name = ort_session.get_inputs()[0].name

    # ONNX Runtime inference - ONE CALL for the entire image!
    ort_outputs = ort_session.run(None, {input_name: three_channel_tensor})
    output = ort_outputs[0]  # (1, 3, H, W)

    # Extract anomaly heatmap (channel 0)
    output_heatmap = output[0, 0, :, :]  # (H, W)

    return output_heatmap, image


def visualize_results(image, heatmap, output_path):
    """Create visualization with all components"""

    # Extract channels from original image
    target = image[:, :, 0]
    ref1 = image[:, :, 1]
    ref2 = image[:, :, 2]

    # Create figure with subplots (7 subplots)
    fig = plt.figure(figsize=(9.5, 6), dpi=200)
    gs = gridspec.GridSpec(1, 7, figure=fig)

    # 1. Target
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(target, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Target')
    ax1.axis('off')

    # 2. Ref1
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(ref1, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Ref1')
    ax2.axis('off')

    # 3. Ref2
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(ref2, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Ref2')
    ax3.axis('off')

    # 4. Target - Ref1
    diff1 = np.abs(target.astype(np.float32) - ref1.astype(np.float32))
    ax4 = fig.add_subplot(gs[3])
    diff1_min = diff1.min()
    diff1_max = diff1.max()
    if diff1_max - diff1_min < 1e-8:
        d1_vmin, d1_vmax = 0, 255
    else:
        d1_vmin, d1_vmax = diff1_min, diff1_max
    ax4.imshow(diff1, cmap='hot', vmin=d1_vmin, vmax=d1_vmax)
    ax4.set_title(f'Target - Ref1')
    ax4.axis('off')

    # 5. Target - Ref2
    diff2 = np.abs(target.astype(np.float32) - ref2.astype(np.float32))
    ax5 = fig.add_subplot(gs[4])
    diff2_min = diff2.min()
    diff2_max = diff2.max()
    if diff2_max - diff2_min < 1e-8:
        d2_vmin, d2_vmax = 0, 255
    else:
        d2_vmin, d2_vmax = diff2_min, diff2_max
    ax5.imshow(diff2, cmap='hot', vmin=d2_vmin, vmax=d2_vmax)
    ax5.set_title(f'Target - Ref2')
    ax5.axis('off')

    # 6. Double Detection
    ax6 = fig.add_subplot(gs[5])
    double_detection = np.minimum(diff1, diff2)
    dd_min = double_detection.min()
    dd_max = double_detection.max()
    if dd_max - dd_min < 1e-8:
        dd_vmin, dd_vmax = 0, 255
    else:
        dd_vmin, dd_vmax = dd_min, dd_max
    ax6.imshow(double_detection, cmap='hot', vmin=dd_vmin, vmax=dd_vmax)
    ax6.set_title('Double Det.')
    ax6.axis('off')

    # 7. Heatmap
    ax7 = fig.add_subplot(gs[6])
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()

    if heatmap_max - heatmap_min < 1e-8:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = heatmap_min, heatmap_max

    im = ax7.imshow(heatmap, cmap='hot', vmin=vmin, vmax=vmax)
    ax7.set_title(f'Heatmap')
    ax7.axis('off')

    # Add colorbar
    plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def inference(args):
    """Main inference function using Full Image ONNX model"""

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load ONNX model
    print(f"Loading ONNX model from: {args.model_path}")

    if not os.path.exists(args.model_path):
        print(f"Error: ONNX model not found: {args.model_path}")
        return

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(args.model_path)

    # Load test images
    image_paths = load_test_images(args.test_path, args.img_format, args.image_type)

    if len(image_paths) == 0:
        print("Warning: No images found!")
        return

    # Extract expected image size from input shape
    input_info = ort_session.get_inputs()[0]
    expected_h = input_info.shape[2]
    expected_w = input_info.shape[3]

    # Storage for AUROC calculation
    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []

    # Inference loop
    for i, img_path in enumerate(image_paths):
        # Load image
        if args.img_format == 'tiff':
            image = tifffile.imread(img_path)
        else:
            image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Failed to load {img_path}")
            continue

        # Convert from CHW to HWC for stripe dataset
        # Support both 3-channel and 4-channel images (4th channel is mask, ignored)
        if args.image_type == 'strip' and (image.shape[0] == 3 or image.shape[0] == 4):
            image = np.transpose(image, (1, 2, 0))

        # Keep only first 3 channels (target, ref1, ref2)
        # If 4-channel image, discard the 4th mask channel
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Normalize to 0-255 range if needed (handles raw sensor values)
        image = image.astype(np.float32)
        img_min = image.min()
        img_max = image.max()
        if img_min < 0 or img_max > 255:
            # Not in 0-255 range, normalize it
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min) * 255.0
            else:
                image = np.zeros_like(image)

        # Verify image size
        h, w = image.shape[:2]
        if h != expected_h or w != expected_w:
            print(f"Warning: Image {os.path.basename(img_path)} has size {h}x{w}, expected {expected_h}x{expected_w}")
            print(f"Skipping this image.")
            continue

        # Perform full image inference (ONE ONNX CALL!)
        heatmap, processed_image = fullimage_inference(image, ort_session, args.image_type)

        # Save visualization (preserve subfolder structure)
        filename = os.path.basename(img_path).split('.')[0]

        # Extract subfolder from image path (e.g., 'bright_spots', 'good')
        relative_path = os.path.relpath(img_path, args.test_path)
        subfolder = os.path.dirname(relative_path)

        # Create output subfolder if needed
        if subfolder:
            output_subfolder = os.path.join(args.output_dir, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)
            output_path = os.path.join(output_subfolder, f'{filename}_result.png')
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f'{filename}_result.png')

        visualize_results(processed_image, heatmap, output_path)
        print(f"Saved result: {output_path}")

        # If using ground truth masks, calculate metrics
        if args.use_ground_truth_mask:
            # Determine if this is a normal or anomaly image
            category = 'good' if 'good' in img_path else 'bright_spots'

            # Get anomaly score
            max_score = heatmap.max()

            if category == 'good':
                # Normal images
                all_scores.append(max_score)
                all_labels.append(0.0)
                pixel_scores.extend(heatmap.flatten())
                pixel_labels.extend(np.zeros_like(heatmap).flatten())
            else:
                # Anomaly images - load ground truth mask
                mask_name = filename + '_mask.' + img_path.split('.')[-1]
                # Navigate from test/bright_spots to ground_truth/bright_spots
                dataset_root = os.path.dirname(os.path.dirname(args.test_path))
                mask_path = os.path.join(dataset_root, 'ground_truth', 'bright_spots', mask_name)

                if os.path.exists(mask_path):
                    # Load mask
                    if args.img_format == 'tiff':
                        gt_mask = tifffile.imread(mask_path)
                        if len(gt_mask.shape) == 3:
                            gt_mask = gt_mask[:, :, 0]
                    else:
                        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    gt_mask = gt_mask.astype(np.float32)
                    if gt_mask.max() > 1:
                        gt_mask = gt_mask / 255.0

                    # Image-level
                    has_anomaly = 1.0 if np.sum(gt_mask) > 0 else 0.0
                    all_scores.append(max_score)
                    all_labels.append(has_anomaly)

                    # Pixel-level
                    pixel_scores.extend(heatmap.flatten())
                    pixel_labels.extend(gt_mask.flatten())
                else:
                    print(f"Warning: Ground truth mask not found: {mask_path}")

    # Calculate and display AUROC if requested
    if args.use_ground_truth_mask and len(all_scores) > 0:
        if len(np.unique(all_labels)) > 1:
            image_auroc = roc_auc_score(all_labels, all_scores)
            print(f"\nImage-level AUROC: {image_auroc:.4f}")
        else:
            print("\nImage-level AUROC: Cannot calculate (only one class present)")

        if len(np.unique(pixel_labels)) > 1:
            pixel_auroc = roc_auc_score(pixel_labels, pixel_scores)
            print(f"Pixel-level AUROC: {pixel_auroc:.4f}")
        else:
            print("Pixel-level AUROC: Cannot calculate (only one class present)")

    print(f"\nInference completed. Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Full Image ONNX Inference for Background Removal Net')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to ONNX model (.onnx file)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output visualizations (default: output)')

    # Optional arguments
    parser.add_argument('--img_format', type=str, choices=['png_jpg', 'tiff'], default='png_jpg',
                        help='Image format (default: png_jpg)')
    parser.add_argument('--image_type', type=str, choices=['strip', 'square'], default='square',
                        help='Image type: strip, square (default: square)')
    parser.add_argument('--use_ground_truth_mask', action='store_true',
                        help='Calculate AUROC using ground truth masks')

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
