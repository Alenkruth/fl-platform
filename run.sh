#!/bin/bash
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        ML Models Demo on RISC-V BOOM (FPGA Emulation)          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Execution Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Platform: RISC-V on FireSim FPGA Emulation"
echo "Python Version: $(/usr/local/bin/python3.10 --version)"
echo "Working Directory: $(pwd)"
echo ""
echo "================================================================"
echo "DEMO OVERVIEW"
echo "================================================================"
echo "This demonstration shows three ML models running on RISC-V:"
echo "  1. Linear Regression"
echo "  2. Deep Neural Network (16 hidden units)"
echo "  3. Logistic Regression"
echo ""
echo "All models show training/test metrics and timing."
echo ""
read -p "Press Enter to begin demonstration..."

# Function to print section headers
print_header() {
    echo ""
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Track total execution time
DEMO_START=$(date +%s)

# Data structure for results
declare -A TRAIN_LOSS TEST_LOSS TRAIN_METRIC TEST_METRIC PARAMS OPTIMIZER TIMES MODEL_NAMES

run_model() {
    local name=$1
    local model=$2
    local opt=$3
    local epochs=$4
    local batch=$5
    local lr=$6
    local idx=$7

    print_header "MODEL ${idx}: ${name}"
    echo "Running: ${model} (${opt})"

    local logfile="run_${model}_${opt}.log"
    local cmd="/usr/local/bin/python3.10 numpy_version/main.py --model ${model} --optimizer ${opt} --epochs ${epochs} --batch ${batch} --lr ${lr}"

    local start=$(date +%s)
    eval "$cmd" | tee "$logfile"
    local end=$(date +%s)
    local runtime=$((end - start))

    # Extract metrics
    TRAIN_LOSS[$idx]=$(grep "Training Loss" "$logfile" | awk '{print $3}')
    TEST_LOSS[$idx]=$(grep "Test Loss" "$logfile" | awk '{print $3}')
    TRAIN_METRIC[$idx]=$(grep "Training R² Score" "$logfile" | awk '{print $4}')
    TEST_METRIC[$idx]=$(grep "Test R² Score" "$logfile" | awk '{print $4}')
    if [ -z "${TRAIN_METRIC[$idx]}" ]; then
        TRAIN_METRIC[$idx]=$(grep "Training Accuracy" "$logfile" | awk '{print $3}')
        TEST_METRIC[$idx]=$(grep "Test Accuracy" "$logfile" | awk '{print $3}')
    fi
    PARAMS[$idx]=$(grep "Model:" "$logfile" -m1 | awk '{print $2}')
    OPTIMIZER[$idx]=$(grep "Optimizer:" "$logfile" -m1 | awk '{print $2}')
    TIMES[$idx]=$runtime
    MODEL_NAMES[$idx]=$name

    echo "✓ ${name} completed in ${runtime}s"
    echo ""
    read -p "Press Enter to continue..."
}

# ============================================================
# Run all models
# ============================================================
run_model "Linear Regression" "least" "sgd" 20 32 0.01 1
run_model "Deep Neural Network" "dnn" "sgd" 20 32 0.01 2
run_model "Logistic Regression" "logistic" "sgd" 20 32 0.01 3
run_model "DNN (SGD Optimizer)" "dnn" "sgd" 30 32 0.01 4
run_model "DNN (SVRG Optimizer)" "dnn" "svrg" 30 32 0.01 5

# ============================================================
# Summary
# ============================================================
DEMO_END=$(date +%s)
TOTAL_TIME=$((DEMO_END - DEMO_START))

echo ""
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     DEMONSTRATION SUMMARY                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
printf "%-30s %-10s %-10s %-15s %-15s %-8s\n" \
    "Model" "TrainLoss" "TestLoss" "TrainAcc/R²" "TestAcc/R²" "Time(s)"
echo "--------------------------------------------------------------------------"

for i in {1..5}; do
    printf "%-30s %-10s %-10s %-15s %-15s %-8s\n" \
        "${MODEL_NAMES[$i]}" \
        "${TRAIN_LOSS[$i]}" \
        "${TEST_LOSS[$i]}" \
        "${TRAIN_METRIC[$i]}" \
        "${TEST_METRIC[$i]}" \
        "${TIMES[$i]}"
done

echo "--------------------------------------------------------------------------"
echo "Total Execution Time: ${TOTAL_TIME}s"
echo "Platform: RISC-V BOOM (FPGA Emulation)"
echo "=========================================================================="
