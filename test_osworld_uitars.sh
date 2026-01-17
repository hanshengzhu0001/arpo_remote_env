#!/bin/bash
# Test OSWorld + UI-TARS-2B Integration

echo "=============================================="
echo "Testing OSWorld + UI-TARS-2B Integration"
echo "=============================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# 1. Check server
if ! curl -s http://localhost:9000/health > /dev/null 2>&1; then
    echo "❌ UI-TARS-2B server not running!"
    echo "   Start it with: python uitars_2b_server.py"
    exit 1
fi
echo "✓ UI-TARS-2B server running"

# 2. Check VMware
if ! vmrun -T fusion list > /dev/null 2>&1; then
    echo "❌ VMware Fusion not configured!"
    echo "   Check: vmrun -T fusion list"
    exit 1
fi
echo "✓ VMware Fusion configured"

# 3. Check conda environment
if [ "$CONDA_DEFAULT_ENV" != "arpo" ]; then
    echo "⚠️  Not in 'arpo' environment"
    echo "   Activating..."
    eval "$(conda shell.bash hook)"
    conda activate arpo
fi
echo "✓ Using 'arpo' environment"

echo ""
echo "Running OSWorld test with UI-TARS-2B..."
echo "This will:"
echo "  - Use VMware VM (Ubuntu)"
echo "  - Run 1 task with max 3 steps"
echo "  - Send screenshots to UI-TARS-2B"
echo "  - Execute predicted actions"
echo ""
echo "Expected time: 1-2 minutes"
echo ""

# Create results directory
mkdir -p results_test_2b/

cd OSWorld

python run_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 3 \
    --model ui-tars-2b \
    --temperature 0.7 \
    --max_tokens 128 \
    --max_trajectory_length 5 \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    --result_dir ../results_test_2b/ \
    2>&1 | tee ../logs/test_osworld_uitars.log

cd ..

echo ""
echo "=============================================="
echo "Test complete!"
echo "=============================================="
echo ""

# Check results
if [ -d "results_test_2b" ]; then
    echo "Results saved to: results_test_2b/"
    echo ""
    echo "View trajectory:"
    find results_test_2b/ -name "traj.jsonl" -exec cat {} \; | head -20
    echo ""
    echo "View result:"
    find results_test_2b/ -name "result.txt" -exec cat {} \;
    echo ""
fi

echo "✓ OSWorld + UI-TARS-2B integration test complete!"
