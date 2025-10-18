import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, base_channels=64):
        super(SegmentationNetwork, self).__init__()
        base_width = base_channels
        self.encoder = EncoderSegmentation(in_channels, base_width)
        self.decoder = DecoderSegmentation(base_width, out_channels=out_channels)
    
    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder(x)
        output = self.decoder(b1, b2, b3, b4, b5, b6)
        return output


class EncoderSegmentation(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderSegmentation, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DecoderSegmentation(nn.Module):
    def __init__(self, base_width, out_channels=2):
        super(DecoderSegmentation, self).__init__()
        
        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        
        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.db_b(cat_b)
        
        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)
        
        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)
        
        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)
        
        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)
        
        out = self.fin_out(db4)
        return out


class SegmentationNetworkONNX(nn.Module):
    """
    ONNX deployment wrapper for SegmentationNetwork

    This wrapper takes the trained SegmentationNetwork and converts its output
    from (B, 2, H, W) to (B, 3, H, W) for production deployment requirements.

    Output channels:
        - Channel 0: Anomaly heatmap (foreground probability after softmax)
        - Channel 1: Zero-filled (placeholder)
        - Channel 2: Zero-filled (placeholder)
    """
    def __init__(self, base_model):
        super(SegmentationNetworkONNX, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # Get base model output: (B, 2, H, W)
        output = self.base_model(x)

        # Apply softmax to get probabilities
        output_sm = F.softmax(output, dim=1)

        # Extract foreground channel (anomaly map)
        anomaly_map = output_sm[:, 1:2, :, :]  # (B, 1, H, W)

        # Create zero channels
        zeros = torch.zeros_like(anomaly_map)

        # Concatenate: [anomaly, zero, zero] -> (B, 3, H, W)
        final_output = torch.cat([anomaly_map, zeros, zeros], dim=1)

        return final_output


class SegmentationNetworkONNXFullImage(nn.Module):
    """
    ONNX deployment wrapper for full image inference with sliding window

    This wrapper processes a complete strip image (976x176) in a single pass:
    - Input: (1, 3, 976, 176) - Full strip image
    - Output: (1, 3, 976, 176) - Full heatmap with 3 channels

    The sliding window logic and patch merging are embedded inside the model.
    This allows deployment as a single ONNX file without external processing.
    """
    def __init__(self, patch_model):
        super(SegmentationNetworkONNXFullImage, self).__init__()
        self.patch_model = patch_model  # SegmentationNetworkONNX instance

        # Fixed configuration for strip images
        self.image_h = 976
        self.image_w = 176
        self.patch_h = 128
        self.patch_w = 128

        # Fixed patch positions (same as training)
        # Y: 9 positions, X: 2 positions
        self.y_positions = torch.tensor([0, 106, 212, 318, 424, 530, 636, 742, 848], dtype=torch.long)
        self.x_positions = torch.tensor([0, 48], dtype=torch.long)

        self.num_y = len(self.y_positions)
        self.num_x = len(self.x_positions)

    def forward(self, x):
        """
        Process full image with sliding window

        Args:
            x: (1, 3, 976, 176) - Full strip image

        Returns:
            (1, 3, 976, 176) - Full heatmap (ch0: anomaly, ch1-2: zeros)
        """
        batch_size = x.shape[0]

        # Initialize output heatmap (only channel 0 will have values)
        output_heatmap = torch.zeros(batch_size, 1, self.image_h, self.image_w,
                                     dtype=x.dtype, device=x.device)

        # Process each patch
        for y_idx in range(self.num_y):
            for x_idx in range(self.num_x):
                y = self.y_positions[y_idx]
                x_pos = self.x_positions[x_idx]

                # Extract patch
                patch = x[:, :, y:y+self.patch_h, x_pos:x_pos+self.patch_w]

                # Inference on patch - returns (1, 3, 128, 128)
                patch_output = self.patch_model(patch)

                # Extract anomaly channel (channel 0)
                patch_heatmap = patch_output[:, 0:1, :, :]  # (1, 1, 128, 128)

                # Determine crop region based on position (strip strategy)
                # Y direction cropping
                if y_idx == 0:
                    # First patch: keep top, remove bottom 11 pixels
                    y_start_crop = 0
                    y_end_crop = self.patch_h - 11
                elif y_idx == self.num_y - 1:
                    # Last patch: remove top 11 pixels, keep bottom
                    y_start_crop = 11
                    y_end_crop = self.patch_h
                else:
                    # Middle patches: remove top and bottom 11 pixels
                    y_start_crop = 11
                    y_end_crop = self.patch_h - 11

                # X direction cropping
                if self.num_x > 1:
                    x_stride = 48  # self.x_positions[1] - self.x_positions[0]
                    x_margin = 40  # (self.patch_w - x_stride) // 2

                    if x_idx == 0:
                        # First patch: keep left, remove right margin
                        x_start_crop = 0
                        x_end_crop = self.patch_w - x_margin
                    elif x_idx == self.num_x - 1:
                        # Last patch: remove left margin, keep right
                        x_start_crop = x_margin
                        x_end_crop = self.patch_w
                    else:
                        # Middle patches (if any)
                        x_start_crop = x_margin
                        x_end_crop = self.patch_w - x_margin
                else:
                    x_start_crop = 0
                    x_end_crop = self.patch_w

                # Extract region to use
                patch_region = patch_heatmap[:, :, y_start_crop:y_end_crop,
                                            x_start_crop:x_end_crop]

                # Calculate output position
                output_y_start = y + y_start_crop
                output_y_end = y + y_end_crop
                output_x_start = x_pos + x_start_crop
                output_x_end = x_pos + x_end_crop

                # Place region in output (direct assignment, no averaging)
                output_heatmap[:, :, output_y_start:output_y_end,
                              output_x_start:output_x_end] = patch_region

        # Create zero channels
        zeros = torch.zeros_like(output_heatmap)

        # Concatenate: [anomaly, zero, zero] -> (1, 3, 976, 176)
        final_output = torch.cat([output_heatmap, zeros, zeros], dim=1)

        return final_output


class SegmentationNetworkONNXSquare(nn.Module):
    """
    ONNX deployment wrapper for square images with sliding window

    This wrapper processes a complete square image (320x320) in a single pass:
    - Input: (1, 3, 320, 320) - Full square image
    - Output: (1, 3, 320, 320) - Full heatmap with 3 channels

    The sliding window logic and patch merging are embedded inside the model.
    """
    def __init__(self, patch_model):
        super(SegmentationNetworkONNXSquare, self).__init__()
        self.patch_model = patch_model  # SegmentationNetworkONNX instance

        # Fixed configuration for square images
        self.image_h = 320
        self.image_w = 320
        self.patch_h = 128
        self.patch_w = 128

        # Fixed patch positions (same as training)
        # Y: 3 positions, X: 3 positions
        self.y_positions = torch.tensor([0, 96, 192], dtype=torch.long)
        self.x_positions = torch.tensor([0, 96, 192], dtype=torch.long)

        self.num_y = len(self.y_positions)
        self.num_x = len(self.x_positions)

    def forward(self, x):
        """
        Process full image with sliding window

        Args:
            x: (1, 3, 320, 320) - Full square image

        Returns:
            (1, 3, 320, 320) - Full heatmap (ch0: anomaly, ch1-2: zeros)
        """
        batch_size = x.shape[0]

        # Initialize output heatmap (only channel 0 will have values)
        output_heatmap = torch.zeros(batch_size, 1, self.image_h, self.image_w,
                                     dtype=x.dtype, device=x.device)

        # Process each patch
        for y_idx in range(self.num_y):
            for x_idx in range(self.num_x):
                y = self.y_positions[y_idx]
                x_pos = self.x_positions[x_idx]

                # Extract patch
                patch = x[:, :, y:y+self.patch_h, x_pos:x_pos+self.patch_w]

                # Inference on patch - returns (1, 3, 128, 128)
                patch_output = self.patch_model(patch)

                # Extract anomaly channel (channel 0)
                patch_heatmap = patch_output[:, 0:1, :, :]  # (1, 1, 128, 128)

                # Determine crop region based on position
                # For 3x3 grid with overlap of 32 pixels, margin is 16 pixels
                margin = 16

                # Y direction cropping
                if y_idx == 0:
                    # First patch: keep top, remove bottom margin
                    y_start_crop = 0
                    y_end_crop = self.patch_h - margin
                elif y_idx == self.num_y - 1:
                    # Last patch: remove top margin, keep bottom
                    y_start_crop = margin
                    y_end_crop = self.patch_h
                else:
                    # Middle patches: remove both margins
                    y_start_crop = margin
                    y_end_crop = self.patch_h - margin

                # X direction cropping
                if x_idx == 0:
                    # First patch: keep left, remove right margin
                    x_start_crop = 0
                    x_end_crop = self.patch_w - margin
                elif x_idx == self.num_x - 1:
                    # Last patch: remove left margin, keep right
                    x_start_crop = margin
                    x_end_crop = self.patch_w
                else:
                    # Middle patches: remove both margins
                    x_start_crop = margin
                    x_end_crop = self.patch_w - margin

                # Extract region to use
                patch_region = patch_heatmap[:, :, y_start_crop:y_end_crop,
                                            x_start_crop:x_end_crop]

                # Calculate output position
                output_y_start = y + y_start_crop
                output_y_end = y + y_end_crop
                output_x_start = x_pos + x_start_crop
                output_x_end = x_pos + x_end_crop

                # Place region in output (direct assignment)
                output_heatmap[:, :, output_y_start:output_y_end,
                              output_x_start:output_x_end] = patch_region

        # Create zero channels
        zeros = torch.zeros_like(output_heatmap)

        # Concatenate: [anomaly, zero, zero] -> (1, 3, 320, 320)
        final_output = torch.cat([output_heatmap, zeros, zeros], dim=1)

        return final_output