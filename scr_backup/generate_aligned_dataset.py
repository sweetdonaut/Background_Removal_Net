import numpy as np
from PIL import Image
from pathlib import Path
import tifffile


def add_gaussian_noise(image, sigma=3):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def crop_with_offset(gray_img, center_x, center_y, crop_width, crop_height, offset_std=5):
    """Crop image with random offset from center"""
    # Generate random offset with normal distribution
    offset_x = int(np.round(np.random.normal(0, offset_std)))
    offset_y = int(np.round(np.random.normal(0, offset_std)))

    # Calculate crop coordinates
    actual_center_x = center_x + offset_x
    actual_center_y = center_y + offset_y

    half_width = crop_width // 2
    half_height = crop_height // 2
    x1 = actual_center_x - half_width
    x2 = actual_center_x + half_width
    y1 = actual_center_y - half_height
    y2 = actual_center_y + half_height

    cropped = gray_img[y1:y2, x1:x2]

    return cropped, offset_x, offset_y


def process_single_image(img_path, crop_width=476, crop_height=468, noise_sigma=3, offset_std=5):
    """Process single RGB image to create 3-channel aligned dataset"""
    # Load and convert to grayscale
    img = Image.open(img_path)
    gray_img = np.array(img.convert('L'))

    height, width = gray_img.shape
    center_x, center_y = width // 2, height // 2

    # Target: crop from center without offset
    half_width = crop_width // 2
    half_height = crop_height // 2
    target = gray_img[
        center_y - half_height:center_y + half_height,
        center_x - half_width:center_x + half_width
    ]

    # Ref1: crop with random offset
    ref1, offset1_x, offset1_y = crop_with_offset(
        gray_img, center_x, center_y, crop_width, crop_height, offset_std
    )

    # Ref2: crop with random offset
    ref2, offset2_x, offset2_y = crop_with_offset(
        gray_img, center_x, center_y, crop_width, crop_height, offset_std
    )

    # Add independent Gaussian noise to each channel
    target_noisy = add_gaussian_noise(target, noise_sigma)
    ref1_noisy = add_gaussian_noise(ref1, noise_sigma)
    ref2_noisy = add_gaussian_noise(ref2, noise_sigma)

    # Stack into 3-channel image (target, ref1, ref2)
    output_img = np.stack([target_noisy, ref1_noisy, ref2_noisy], axis=2)

    return output_img, (offset1_x, offset1_y), (offset2_x, offset2_y)


def main():
    # Paths
    input_base = Path("/home/yclai/vscode_project/Background_Removal_Net/MVTec_AD_dataset/grid_offset")
    output_base = Path("/home/yclai/vscode_project/Background_Removal_Net/MVTec_AD_dataset/grid_offset_3channel")

    # Parameters
    crop_width = 476
    crop_height = 468
    noise_sigma = 3
    offset_std = 5

    # Find all PNG files
    png_files = sorted(input_base.rglob("*.png"))

    print(f"Found {len(png_files)} images to process")
    print(f"Parameters:")
    print(f"  - Crop size: {crop_width}x{crop_height}")
    print(f"  - Noise sigma: {noise_sigma}")
    print(f"  - Offset std: {offset_std}")
    print()

    # Process each image
    for i, img_path in enumerate(png_files):
        # Calculate relative path to maintain folder structure
        rel_path = img_path.relative_to(input_base)
        output_path = output_base / rel_path.parent / f"{rel_path.stem}.tiff"

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process image
        output_img, offset1, offset2 = process_single_image(
            img_path, crop_width, crop_height, noise_sigma, offset_std
        )

        # Save as TIFF
        tifffile.imwrite(output_path, output_img)

        if (i + 1) % 10 == 0 or (i + 1) == len(png_files):
            print(f"Processed {i + 1}/{len(png_files)}: {rel_path}")
            print(f"  Ref1 offset: ({offset1[0]:+3d}, {offset1[1]:+3d}) pixels")
            print(f"  Ref2 offset: ({offset2[0]:+3d}, {offset2[1]:+3d}) pixels")

    print(f"\nDone! Output saved to: {output_base}")


if __name__ == "__main__":
    main()
