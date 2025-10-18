import numpy as np
import cv2
import tifffile
from pathlib import Path
from scipy import ndimage


def align_template_matching(target, ref, search_range=20):
    """
    Align ref to target using template matching

    Args:
        target: Target grayscale image
        ref: Reference grayscale image to align
        search_range: Search range in pixels (default: 20)

    Returns:
        (shift_x, shift_y, aligned_ref)
    """
    h, w = target.shape

    # Use center region as template to avoid edge effects
    margin = search_range + 10
    template = target[margin:-margin, margin:-margin]

    # Search in ref image
    result = cv2.matchTemplate(ref, template, cv2.TM_CCOEFF_NORMED)

    # Find best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Calculate shift
    expected_x = margin
    expected_y = margin
    actual_x, actual_y = max_loc

    shift_x = expected_x - actual_x
    shift_y = expected_y - actual_y

    # Apply shift to ref
    aligned_ref = ndimage.shift(ref, shift=(shift_y, shift_x), mode='nearest')

    return shift_x, shift_y, aligned_ref


def part1_alignment(input_path, output_path):
    """
    Part 1: Align three channels and save

    Args:
        input_path: Path to input TIFF file
        output_path: Path to output aligned TIFF file

    Returns:
        dict with alignment info
    """
    # Read TIFF
    img = tifffile.imread(input_path)

    # Extract channels
    target = img[:, :, 0]
    ref1 = img[:, :, 1]
    ref2 = img[:, :, 2]

    # Align ref1 to target
    shift1_x, shift1_y, aligned_ref1 = align_template_matching(target, ref1)

    # Align ref2 to target
    shift2_x, shift2_y, aligned_ref2 = align_template_matching(target, ref2)

    # Stack aligned channels
    aligned_img = np.stack([target, aligned_ref1, aligned_ref2], axis=2)

    # Save aligned TIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, aligned_img.astype(np.uint8))

    # Return alignment info
    return {
        'ref1_shift': (shift1_x, shift1_y),
        'ref2_shift': (shift2_x, shift2_y),
        'shape': aligned_img.shape
    }


def part2_crop(input_path, output_path, crop_size=320):
    """
    Part 2: Crop aligned image to smaller size

    Args:
        input_path: Path to aligned TIFF file
        output_path: Path to output cropped TIFF file
        crop_size: Size to crop (default: 320)

    Returns:
        dict with crop info
    """
    # Read aligned TIFF
    img = tifffile.imread(input_path)

    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2

    # Calculate crop region
    half_size = crop_size // 2
    y1 = center_y - half_size
    y2 = center_y + half_size
    x1 = center_x - half_size
    x2 = center_x + half_size

    # Crop
    cropped_img = img[y1:y2, x1:x2, :]

    # Save cropped TIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, cropped_img)

    return {
        'original_shape': img.shape,
        'cropped_shape': cropped_img.shape,
        'crop_region': (x1, y1, x2, y2)
    }


def main():
    # Configuration
    CROP_SIZE = 320

    # Paths
    input_base = Path("/home/yclai/vscode_project/Background_Removal_Net/MVTec_AD_dataset/grid_offset_3channel")
    output_aligned_base = Path("/home/yclai/vscode_project/Background_Removal_Net/output/aligned_template_matching")
    output_cropped_base = Path("/home/yclai/vscode_project/Background_Removal_Net/output/aligned_cropped_template_matching")

    # Find all TIFF files
    tiff_files = sorted(input_base.rglob("*.tiff"))

    print("=" * 80)
    print("Image Alignment Pipeline (Template Matching)")
    print("=" * 80)
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"Found {len(tiff_files)} images to process")
    print()

    # Process each image
    for i, tiff_path in enumerate(tiff_files):
        rel_path = tiff_path.relative_to(input_base)

        # Part 1: Alignment
        aligned_path = output_aligned_base / rel_path
        align_info = part1_alignment(tiff_path, aligned_path)

        # Part 2: Crop
        cropped_path = output_cropped_base / rel_path
        crop_info = part2_crop(aligned_path, cropped_path, crop_size=CROP_SIZE)

        # Print progress
        if (i + 1) % 5 == 0 or (i + 1) == len(tiff_files):
            print(f"Processed {i + 1}/{len(tiff_files)}: {rel_path}")
            print(f"  Ref1 shift: ({align_info['ref1_shift'][0]:+7.2f}, {align_info['ref1_shift'][1]:+7.2f}) pixels")
            print(f"  Ref2 shift: ({align_info['ref2_shift'][0]:+7.2f}, {align_info['ref2_shift'][1]:+7.2f}) pixels")
            print(f"  Cropped: {crop_info['original_shape'][:2]} -> {crop_info['cropped_shape'][:2]}")

    print()
    print("=" * 80)
    print("Pipeline completed!")
    print(f"Aligned images (384x384): {output_aligned_base}")
    print(f"Cropped images (320x320): {output_cropped_base}")
    print("=" * 80)


if __name__ == "__main__":
    main()
