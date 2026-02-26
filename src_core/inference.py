import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import argparse
import glob
import tifffile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import SegmentationNetwork
from dataloader import calculate_positions, ensure_hwc, ensure_3ch


class InferenceDataset(Dataset):

    def __init__(self, test_path, img_format='png_jpg'):
        self.img_format = img_format

        self.image_paths = []
        if img_format == 'png_jpg':
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.png")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.jpg")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.png")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.jpg")))
        else:
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.tiff")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*.tif")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.tiff")))
            self.image_paths.extend(glob.glob(os.path.join(test_path, "*", "*.tif")))

        self.image_paths = sorted(self.image_paths)
        print(f"Found {len(self.image_paths)} test images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        if self.img_format == 'tiff':
            image = tifffile.imread(img_path)
        else:
            image = cv2.imread(img_path)

        image = ensure_hwc(image)
        image = ensure_3ch(image)

        image = image.astype(np.float32)
        img_min, img_max = image.min(), image.max()
        if img_min < 0 or img_max > 255:
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min) * 255.0
            else:
                image = np.zeros_like(image)

        original_h, original_w = image.shape[:2]

        return {
            'image': image,
            'image_path': img_path,
            'original_size': (original_h, original_w)
        }


def sliding_window_inference(image, model, patch_size, device):
    h, w = image.shape[:2]
    patch_h, patch_w = patch_size

    if h < patch_h or w < patch_w:
        raise ValueError(f"Image size ({h}x{w}) is smaller than patch size ({patch_h}x{patch_w}).")

    y_positions = calculate_positions(h, patch_h)
    x_positions = calculate_positions(w, patch_w)

    if y_positions is None or x_positions is None:
        raise ValueError(f"Image size ({h}x{w}) is too small for patch size ({patch_h}x{patch_w})")

    output_heatmap = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for y_idx, y in enumerate(y_positions):
        for x_idx, x in enumerate(x_positions):
            patch = image[y:y+patch_h, x:x+patch_w]

            three_channel = np.stack([patch[:,:,0], patch[:,:,1], patch[:,:,2]], axis=0)
            three_channel_tensor = torch.from_numpy(three_channel).float() / 255.0
            three_channel_tensor = three_channel_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(three_channel_tensor)
                output_sm = F.softmax(output, dim=1)
                patch_heatmap = output_sm[:, 1, :, :].squeeze().cpu().numpy()

            if len(y_positions) > 1 or len(x_positions) > 1:
                # Center-crop stitching: each patch only contributes its non-overlapping center
                y_stride = y_positions[1] - y_positions[0] if len(y_positions) > 1 else patch_h
                y_margin = (patch_h - y_stride) // 2

                if y_idx == 0:
                    y_start_crop, y_end_crop = 0, patch_h - y_margin
                elif y_idx == len(y_positions) - 1:
                    y_start_crop, y_end_crop = y_margin, patch_h
                else:
                    y_start_crop, y_end_crop = y_margin, patch_h - y_margin

                if len(x_positions) > 1:
                    x_stride = x_positions[1] - x_positions[0]
                    x_margin = (patch_w - x_stride) // 2

                    if x_idx == 0:
                        x_start_crop, x_end_crop = 0, patch_w - x_margin
                    elif x_idx == len(x_positions) - 1:
                        x_start_crop, x_end_crop = x_margin, patch_w
                    else:
                        x_start_crop, x_end_crop = x_margin, patch_w - x_margin
                else:
                    x_start_crop, x_end_crop = 0, patch_w

                patch_region = patch_heatmap[y_start_crop:y_end_crop, x_start_crop:x_end_crop]

                oy_s, oy_e = y + y_start_crop, y + y_end_crop
                ox_s, ox_e = x + x_start_crop, x + x_end_crop

                output_heatmap[oy_s:oy_e, ox_s:ox_e] = patch_region
                weight_map[oy_s:oy_e, ox_s:ox_e] = 1
            else:
                output_heatmap[y:y+patch_h, x:x+patch_w] = patch_heatmap
                weight_map[y:y+patch_h, x:x+patch_w] = 1

    output_heatmap = output_heatmap / np.maximum(weight_map, 1)
    return output_heatmap, image


def visualize_results(image, heatmap, output_path):
    target = image[:, :, 0]
    ref1 = image[:, :, 1]
    ref2 = image[:, :, 2]

    fig = plt.figure(figsize=(9.5, 6), dpi=200)
    gs = gridspec.GridSpec(1, 7, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(target, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Target')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(ref1, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Ref1')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(ref2, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Ref2')
    ax3.axis('off')

    diff1 = np.abs(target.astype(np.float32) - ref1.astype(np.float32))
    ax4 = fig.add_subplot(gs[3])
    d1_min, d1_max = diff1.min(), diff1.max()
    if d1_max - d1_min < 1e-8:
        d1_min, d1_max = 0, 255
    ax4.imshow(diff1, cmap='hot', vmin=d1_min, vmax=d1_max)
    ax4.set_title('Target - Ref1')
    ax4.axis('off')

    diff2 = np.abs(target.astype(np.float32) - ref2.astype(np.float32))
    ax5 = fig.add_subplot(gs[4])
    d2_min, d2_max = diff2.min(), diff2.max()
    if d2_max - d2_min < 1e-8:
        d2_min, d2_max = 0, 255
    ax5.imshow(diff2, cmap='hot', vmin=d2_min, vmax=d2_max)
    ax5.set_title('Target - Ref2')
    ax5.axis('off')

    double_detection = np.minimum(diff1, diff2)
    ax6 = fig.add_subplot(gs[5])
    dd_min, dd_max = double_detection.min(), double_detection.max()
    if dd_max - dd_min < 1e-8:
        dd_min, dd_max = 0, 255
    ax6.imshow(double_detection, cmap='hot', vmin=dd_min, vmax=dd_max)
    ax6.set_title('Double Det.')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[6])
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-8:
        h_min, h_max = 0, 1
    im = ax7.imshow(heatmap, cmap='hot', vmin=h_min, vmax=h_max)
    ax7.set_title('Heatmap')
    ax7.axis('off')

    plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def inference(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    patch_size = (checkpoint['img_height'], checkpoint['img_width'])
    print(f"Model patch size: {patch_size}")

    model = SegmentationNetwork(in_channels=3, out_channels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    dataset = InferenceDataset(
        test_path=args.test_path,
        img_format=args.img_format,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []

    for i, sample in enumerate(dataloader):
        image = sample['image'].squeeze().numpy()
        img_path = sample['image_path'][0]

        heatmap, processed_image = sliding_window_inference(
            image, model, patch_size, device
        )

        filename = os.path.basename(img_path).split('.')[0]
        relative_path = os.path.relpath(img_path, args.test_path)
        subfolder = os.path.dirname(relative_path)

        if subfolder:
            output_subfolder = os.path.join(args.output_dir, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)
            output_path = os.path.join(output_subfolder, f'{filename}_result.png')
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, f'{filename}_result.png')

        visualize_results(processed_image, heatmap, output_path)
        print(f"Saved result: {output_path}")

        if args.use_ground_truth_mask:
            category = 'good' if 'good' in img_path else 'bright_spots'
            max_score = heatmap.max()

            if category == 'good':
                all_scores.append(max_score)
                all_labels.append(0.0)
                pixel_scores.extend(heatmap.flatten())
                pixel_labels.extend(np.zeros_like(heatmap).flatten())
            else:
                mask_name = filename + '_mask.' + img_path.split('.')[-1]
                dataset_root = os.path.dirname(os.path.dirname(args.test_path))
                mask_path = os.path.join(dataset_root, 'ground_truth', 'bright_spots', mask_name)

                if os.path.exists(mask_path):
                    if args.img_format == 'tiff':
                        gt_mask = tifffile.imread(mask_path)
                        if len(gt_mask.shape) == 3:
                            gt_mask = gt_mask[:, :, 0]
                    else:
                        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    gt_mask = gt_mask.astype(np.float32)

                    has_anomaly = 1.0 if np.sum(gt_mask) > 0 else 0.0
                    all_scores.append(max_score)
                    all_labels.append(has_anomaly)

                    pixel_scores.extend(heatmap.flatten())
                    pixel_labels.extend(gt_mask.flatten())
                else:
                    print(f"Warning: Ground truth mask not found: {mask_path}")

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
    parser = argparse.ArgumentParser(description='Inference for Background Removal Net')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output visualizations')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use. Set to -1 for CPU (default: 0)')
    parser.add_argument('--img_format', type=str, choices=['png_jpg', 'tiff'], default='png_jpg',
                        help='Image format (default: png_jpg)')
    parser.add_argument('--use_ground_truth_mask', type=str, choices=['True', 'False'], default='False',
                        help='Whether to calculate AUROC using ground truth masks (default: False)')

    args = parser.parse_args()
    args.use_ground_truth_mask = args.use_ground_truth_mask == 'True'

    inference(args)


if __name__ == "__main__":
    main()
