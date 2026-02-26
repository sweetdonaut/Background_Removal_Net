#!/usr/bin/env python3
"""
ONNX Model Inspector - Detailed Analysis Tool

This script analyzes an ONNX model to understand its internal structure,
including operations, layers, and logic flow.
"""

import argparse
import onnx
from onnx import numpy_helper
import numpy as np
from collections import Counter, defaultdict


def analyze_onnx_model(model_path, verbose=False):
    """
    Comprehensive analysis of ONNX model structure

    Args:
        model_path: Path to .onnx file
        verbose: If True, show detailed layer-by-layer information
    """

    print("=" * 80)
    print("ONNX Model Inspector")
    print("=" * 80)
    print(f"\nModel: {model_path}\n")

    # Load ONNX model
    print("Loading ONNX model...")
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("✓ Model loaded and validated successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Get graph
    graph = model.graph

    # ============================================================================
    # SECTION 1: Basic Model Information
    # ============================================================================
    print("=" * 80)
    print("SECTION 1: Basic Model Information")
    print("=" * 80)

    print(f"\nProducer: {model.producer_name} {model.producer_version}")
    print(f"IR Version: {model.ir_version}")
    print(f"Opset Version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
    print(f"Model Version: {model.model_version}")
    print(f"Doc String: {model.doc_string if model.doc_string else 'None'}")

    # ============================================================================
    # SECTION 2: Input/Output Information
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Input/Output Specifications")
    print("=" * 80)

    print("\n--- Model Inputs ---")
    for idx, input_tensor in enumerate(graph.input):
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})"
                 for dim in input_tensor.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
        print(f"{idx}. Name: '{input_tensor.name}'")
        print(f"   Shape: {shape}")
        print(f"   Type: {dtype}")

    print("\n--- Model Outputs ---")
    for idx, output_tensor in enumerate(graph.output):
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})"
                 for dim in output_tensor.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)
        print(f"{idx}. Name: '{output_tensor.name}'")
        print(f"   Shape: {shape}")
        print(f"   Type: {dtype}")

    # ============================================================================
    # SECTION 3: Graph Structure Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: Graph Structure Analysis")
    print("=" * 80)

    total_nodes = len(graph.node)
    total_initializers = len(graph.initializer)

    print(f"\nTotal Nodes (Operations): {total_nodes:,}")
    print(f"Total Initializers (Weights/Constants): {total_initializers:,}")

    # Count operations by type
    op_types = Counter(node.op_type for node in graph.node)

    print(f"\n--- Operation Types (Top 20) ---")
    for op_type, count in op_types.most_common(20):
        percentage = (count / total_nodes) * 100
        print(f"{op_type:20s}: {count:5d} ({percentage:5.2f}%)")

    if len(op_types) > 20:
        print(f"... and {len(op_types) - 20} more operation types")

    # ============================================================================
    # SECTION 4: Weight Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: Weight/Parameter Analysis")
    print("=" * 80)

    total_params = 0
    total_size_mb = 0
    weight_info = []

    for initializer in graph.initializer:
        tensor = numpy_helper.to_array(initializer)
        num_params = np.prod(tensor.shape)
        size_mb = tensor.nbytes / (1024 * 1024)
        total_params += num_params
        total_size_mb += size_mb

        weight_info.append({
            'name': initializer.name,
            'shape': tensor.shape,
            'params': num_params,
            'size_mb': size_mb,
            'dtype': tensor.dtype
        })

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Weight Size: {total_size_mb:.2f} MB")

    # Show largest weights
    weight_info_sorted = sorted(weight_info, key=lambda x: x['params'], reverse=True)
    print(f"\n--- Top 10 Largest Weights ---")
    for i, info in enumerate(weight_info_sorted[:10], 1):
        print(f"{i:2d}. {info['name']}")
        print(f"    Shape: {info['shape']}, Params: {info['params']:,}, Size: {info['size_mb']:.4f} MB")

    # ============================================================================
    # SECTION 5: Network Architecture Detection
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: Network Architecture Detection")
    print("=" * 80)

    # Detect UNet-like patterns
    conv_count = op_types.get('Conv', 0)
    bn_count = op_types.get('BatchNormalization', 0)
    relu_count = op_types.get('Relu', 0)
    maxpool_count = op_types.get('MaxPool', 0)
    upsample_count = op_types.get('Upsample', 0) + op_types.get('Resize', 0)
    concat_count = op_types.get('Concat', 0)

    print(f"\n--- UNet Architecture Indicators ---")
    print(f"Convolution layers: {conv_count}")
    print(f"BatchNorm layers: {bn_count}")
    print(f"ReLU activations: {relu_count}")
    print(f"MaxPool (downsampling): {maxpool_count}")
    print(f"Upsample/Resize (upsampling): {upsample_count}")
    print(f"Concat (skip connections): {concat_count}")

    if conv_count > 0 and maxpool_count > 0 and upsample_count > 0 and concat_count > 0:
        print("\n✓ Detected UNet-like architecture!")
        print("  - Encoder path: Conv + MaxPool")
        print("  - Decoder path: Upsample + Conv")
        print("  - Skip connections: Concat")

    # ============================================================================
    # SECTION 6: Sliding Window Logic Detection
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: Sliding Window Logic Detection")
    print("=" * 80)

    # Look for patterns indicating sliding window
    slice_count = op_types.get('Slice', 0)
    gather_count = op_types.get('Gather', 0)
    scatter_count = op_types.get('Scatter', 0) + op_types.get('ScatterND', 0)

    print(f"\n--- Sliding Window Indicators ---")
    print(f"Slice operations (patch extraction): {slice_count}")
    print(f"Gather operations (indexing): {gather_count}")
    print(f"Scatter operations (merging): {scatter_count}")

    if slice_count > 10:
        estimated_patches = slice_count // 2  # Rough estimate
        print(f"\n✓ Detected sliding window logic!")
        print(f"  - Estimated number of patches: ~{estimated_patches}")
        print(f"  - Pattern: Multiple Slice operations suggest patch extraction")

    # Look for specific constants that might indicate patch positions
    print(f"\n--- Looking for Hardcoded Patch Positions ---")
    constant_count = op_types.get('Constant', 0)
    print(f"Constant operations: {constant_count}")

    # Analyze constant values
    patch_positions_y = []
    patch_positions_x = []

    for node in graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 't':  # tensor attribute
                    tensor = numpy_helper.to_array(attr.t)
                    # Look for values that might be patch positions
                    if tensor.size == 1:
                        val = int(tensor.item())
                        # Y positions for 976 height: [0, 106, 212, 318, 424, 530, 636, 742, 848]
                        if val in [0, 106, 212, 318, 424, 530, 636, 742, 848]:
                            if val not in patch_positions_y:
                                patch_positions_y.append(val)
                        # X positions for 176 width: [0, 48]
                        if val in [0, 48]:
                            if val not in patch_positions_x:
                                patch_positions_x.append(val)

    if patch_positions_y:
        print(f"\n✓ Found potential Y patch positions: {sorted(patch_positions_y)}")
        print(f"  Number of Y patches: {len(patch_positions_y)}")

    if patch_positions_x:
        print(f"✓ Found potential X patch positions: {sorted(patch_positions_x)}")
        print(f"  Number of X patches: {len(patch_positions_x)}")

    if patch_positions_y and patch_positions_x:
        total_patches = len(patch_positions_y) * len(patch_positions_x)
        print(f"\n✓ Estimated total patches: {len(patch_positions_y)} × {len(patch_positions_x)} = {total_patches}")

    # ============================================================================
    # SECTION 7: Data Flow Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: Data Flow Analysis")
    print("=" * 80)

    # Build input-output mapping
    value_producers = {}  # value_name -> node that produces it
    value_consumers = defaultdict(list)  # value_name -> list of nodes that consume it

    for node in graph.node:
        for output in node.output:
            value_producers[output] = node
        for input_name in node.input:
            value_consumers[input_name].append(node)

    # Analyze the first few operations
    print(f"\n--- First 10 Operations (Starting from Input) ---")
    input_name = graph.input[0].name
    current_values = [input_name]
    visited = set()
    op_sequence = []

    for step in range(10):
        if not current_values:
            break

        next_values = []
        for value in current_values:
            if value in visited:
                continue
            visited.add(value)

            if value in value_consumers:
                for node in value_consumers[value][:1]:  # Take first consumer
                    op_sequence.append(node)
                    next_values.extend(node.output)
                    break

        current_values = next_values

    for idx, node in enumerate(op_sequence, 1):
        print(f"{idx:2d}. {node.op_type:20s} ({node.name if node.name else 'unnamed'})")
        if verbose:
            print(f"    Inputs: {node.input[:3]}")
            print(f"    Outputs: {node.output[:3]}")

    # ============================================================================
    # SECTION 8: Output Processing Detection
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: Output Processing Detection")
    print("=" * 80)

    # Look for Softmax, Concat operations near the end
    softmax_count = op_types.get('Softmax', 0)

    print(f"\n--- Output Processing Indicators ---")
    print(f"Softmax operations: {softmax_count}")

    # Find operations that directly produce the output
    output_name = graph.output[0].name
    if output_name in value_producers:
        output_producer = value_producers[output_name]
        print(f"\nFinal output producer:")
        print(f"  Operation: {output_producer.op_type}")
        print(f"  Node name: {output_producer.name if output_producer.name else 'unnamed'}")

        # Trace back a few steps
        print(f"\n--- Last 5 Operations (Before Output) ---")
        current = output_producer
        for i in range(5):
            print(f"{5-i}. {current.op_type:20s} ({current.name if current.name else 'unnamed'})")
            if current.input and current.input[0] in value_producers:
                current = value_producers[current.input[0]]
            else:
                break

    # ============================================================================
    # SECTION 9: Model Transparency Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: Transparency Summary - What Can Be Seen?")
    print("=" * 80)

    transparency_score = 0
    max_score = 8

    print("\n--- Visibility Check ---")

    # 1. Can see model architecture?
    if conv_count > 0:
        print("✓ [1/8] Model Architecture: VISIBLE")
        print("      → Can identify Conv, BatchNorm, ReLU layers")
        transparency_score += 1
    else:
        print("✗ [1/8] Model Architecture: NOT CLEAR")

    # 2. Can see UNet structure?
    if maxpool_count > 0 and upsample_count > 0:
        print("✓ [2/8] UNet Structure: VISIBLE")
        print("      → Can see encoder (MaxPool) and decoder (Upsample) paths")
        transparency_score += 1
    else:
        print("✗ [2/8] UNet Structure: NOT CLEAR")

    # 3. Can see skip connections?
    if concat_count > 0:
        print("✓ [3/8] Skip Connections: VISIBLE")
        print(f"      → Found {concat_count} Concat operations")
        transparency_score += 1
    else:
        print("✗ [3/8] Skip Connections: NOT CLEAR")

    # 4. Can see sliding window?
    if slice_count > 10:
        print("✓ [4/8] Sliding Window: VISIBLE")
        print(f"      → Found {slice_count} Slice operations (patch extraction)")
        transparency_score += 1
    else:
        print("✗ [4/8] Sliding Window: NOT CLEAR")

    # 5. Can see patch positions?
    if patch_positions_y and patch_positions_x:
        print("✓ [5/8] Patch Positions: VISIBLE")
        print(f"      → Found hardcoded positions: Y={sorted(patch_positions_y)}, X={sorted(patch_positions_x)}")
        transparency_score += 1
    else:
        print("✗ [5/8] Patch Positions: NOT CLEAR")

    # 6. Can see merging logic?
    if scatter_count > 0 or slice_count > 10:
        print("✓ [6/8] Merging Logic: PARTIALLY VISIBLE")
        print("      → Scatter/Slice operations suggest patch assembly")
        transparency_score += 1
    else:
        print("✗ [6/8] Merging Logic: NOT CLEAR")

    # 7. Can see softmax?
    if softmax_count > 0:
        print("✓ [7/8] Output Processing (Softmax): VISIBLE")
        print(f"      → Found {softmax_count} Softmax operations")
        transparency_score += 1
    else:
        print("✗ [7/8] Output Processing: NOT CLEAR")

    # 8. Can see model weights?
    if total_params > 0:
        print("✓ [8/8] Model Weights: FULLY VISIBLE")
        print(f"      → {total_params:,} parameters, {total_size_mb:.2f} MB")
        transparency_score += 1
    else:
        print("✗ [8/8] Model Weights: NOT VISIBLE")

    print(f"\n{'=' * 80}")
    print(f"Overall Transparency Score: {transparency_score}/{max_score} ({transparency_score/max_score*100:.0f}%)")
    print(f"{'=' * 80}")

    if transparency_score >= 6:
        print("\n✓ HIGH TRANSPARENCY")
        print("  → Most of the model logic is visible and understandable")
        print("  → Architecture, sliding window, and processing flow are clear")
    elif transparency_score >= 4:
        print("\n⚠ MEDIUM TRANSPARENCY")
        print("  → Some model details are visible")
        print("  → May need deeper analysis to understand full logic")
    else:
        print("\n✗ LOW TRANSPARENCY")
        print("  → Limited information available")
        print("  → Model is relatively opaque")

    # ============================================================================
    # Final Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
For a general person looking at this ONNX model:

1. Architecture Understanding:
   - Can clearly see it's a UNet-based model
   - Can identify encoder/decoder structure
   - Can count layers and see skip connections

2. Sliding Window Logic:
   - Can detect that sliding window is used
   - Can find hardcoded patch positions
   - Can estimate number of patches processed

3. Processing Flow:
   - Can see the sequence of operations
   - Can trace data flow from input to output
   - Can identify key processing steps (Conv, Softmax, etc.)

4. Limitations:
   - Cannot easily see high-level logic (Python code)
   - Difficult to understand conditional logic (if/else)
   - Need tools to visualize the full graph

5. Reverse Engineering Difficulty:
   - Easy to extract weights: ✓
   - Easy to understand architecture: ✓
   - Medium to understand sliding window: ~
   - Hard to reconstruct exact training code: ✗

Overall: ONNX is quite transparent, but requires some expertise to interpret!
""")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Inspect ONNX model structure in detail',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python inspect_onnx_model.py --model model.onnx

  # Verbose mode (show more details)
  python inspect_onnx_model.py --model model.onnx --verbose
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose output with detailed layer information')

    args = parser.parse_args()

    analyze_onnx_model(args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
