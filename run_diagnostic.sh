#!/bin/bash

echo "Running diagnostic test for stripe/edge detection issues..."
echo "============================================"

python diagnostic_test.py

echo ""
echo "Diagnostic test completed!"
echo "Please check the results in ./output/diagnostic_test/"
echo ""
echo "Key files to review:"
echo "  - summary.png: Overview of all test cases"
echo "  - 2_identical_channels_viz.png: Should show NO response"
echo "  - 6_dots_in_target_viz.png: Should show strong dot detection"
echo "  - 7_edge_enhancement_viz.png: Should NOT show edge as defects"