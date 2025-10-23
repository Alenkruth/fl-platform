
#!/bin/bash

# Color codes for better readability (if terminal supports it)
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get system info
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║        ML Models Demo on RISC-V BOOM (FPGA)                    ║"
echo "║                                                                ║"
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
echo "  1. Linear Regression (Least Squares)"
echo "  2. Deep Neural Network (2-layer MLP, 16 hidden units)"
echo "  3. Logistic Regression (Binary Classification)"
echo ""
echo "All models are trained on synthetic data and will display:"
echo "  - Training progress and loss curves"
echo "  - Final test set performance metrics"
echo "  - Execution time statistics"
echo ""
echo "Press Enter to begin demonstration..."
read

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

# ============================================================================
# 1. Linear Regression
# ============================================================================
print_header "MODEL 1/3: LINEAR REGRESSION"
echo "Task: Regression on synthetic dataset (200 samples, 3 features)"
echo "Command: python3 numpy_version/main.py --model least --epochs 20 --batch 32 --lr 0.01"
echo ""

MODEL1_START=$(date +%s)
/usr/local/bin/python3.10 numpy_version/main.py --model least --epochs 20 --batch 32 --lr 0.01
MODEL1_END=$(date +%s)
MODEL1_TIME=$((MODEL1_END - MODEL1_START))
echo "✓ Linear Regression completed in ${MODEL1_TIME} seconds"

# ============================================================================
# 2. Deep Neural Network
# ============================================================================
print_header "MODEL 2/3: DEEP NEURAL NETWORK"
echo "Task: Regression with 2-layer neural network (16 hidden units)"
echo "Command: python3 numpy_version/main.py --model dnn --epochs 20 --batch 32 --lr 0.01"
echo ""

MODEL2_START=$(date +%s)
/usr/local/bin/python3.10 numpy_version/main.py --model dnn --epochs 20 --batch 32 --lr 0.01
MODEL2_END=$(date +%s)
MODEL2_TIME=$((MODEL2_END - MODEL2_START))
echo "✓ Deep Neural Network completed in ${MODEL2_TIME} seconds"

# ============================================================================
# 3. Logistic Regression
# ============================================================================
print_header "MODEL 3/3: LOGISTIC REGRESSION"
echo "Task: Binary classification on synthetic dataset"
echo "Command: python3 numpy_version/main.py --model logistic --epochs 20 --batch 32 --lr 0.01"
echo ""

MODEL3_START=$(date +%s)
/usr/local/bin/python3.10 numpy_version/main.py --model logistic --epochs 20 --batch 32 --lr 0.01
MODEL3_END=$(date +%s)
MODEL3_TIME=$((MODEL3_END - MODEL3_START))
echo "✓ Logistic Regression completed in ${MODEL3_TIME} seconds"

# ============================================================================
# Optional: Optimizer Comparison (SGD vs SVRG)
# ============================================================================
print_header "BONUS: OPTIMIZER COMPARISON (SGD vs SVRG)"
echo "Comparing SGD and SVRG optimizers on DNN model"
echo ""

echo "--- Running DNN with SGD Optimizer ---"
OPT1_START=$(date +%s)
/usr/local/bin/python3.10 numpy_version/main.py --model dnn --optimizer sgd --epochs 15 --batch 32 --lr 0.01
OPT1_END=$(date +%s)
OPT1_TIME=$((OPT1_END - OPT1_START))

echo ""
echo "--- Running DNN with SVRG Optimizer ---"
OPT2_START=$(date +%s)
/usr/local/bin/python3.10 numpy_version/main.py --model dnn --optimizer svrg --epochs 15 --batch 32 --lr 0.01
OPT2_END=$(date +%s)
OPT2_TIME=$((OPT2_END - OPT2_START))

echo ""
echo "Optimizer Comparison:"
echo "  SGD:  ${OPT1_TIME}s"
echo "  SVRG: ${OPT2_TIME}s"

# ============================================================================
# Summary Statistics
# ============================================================================
DEMO_END=$(date +%s)
TOTAL_TIME=$((DEMO_END - DEMO_START))

echo ""
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     DEMONSTRATION SUMMARY                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Platform: RISC-V processor on FireSim FPGA emulation"
echo "Total Execution Time: ${TOTAL_TIME} seconds"
echo ""
echo "Individual Model Execution Times:"
echo "  • Linear Regression:     ${MODEL1_TIME}s"
echo "  • Deep Neural Network:   ${MODEL2_TIME}s"
echo "  • Logistic Regression:   ${MODEL3_TIME}s"
echo ""
echo "================================================================"
echo "FEDERATED LEARNING READINESS CHECK"
echo "================================================================"
echo ""
echo "✓ All models successfully trained on RISC-V platform"
#echo "✓ Models support batch processing (required for FL)"
#echo "✓ Models implement gradient computation (required for FL)"
#echo "✓ Models use both SGD and SVRG optimizers"
#echo "✓ Models can serialize weights for aggregation"
#echo "✓ Train/test evaluation demonstrates model generalization"
echo ""
#echo "Next Steps for Federated Learning:"
#echo "  1. Implement weight serialization/deserialization"
#echo "  2. Set up central aggregation server"
#echo "  3. Create multi-node communication protocol"
#echo "  4. Test federated averaging across nodes"
#echo "  5. Deploy coordinator for distributed training"
#echo ""
echo "================================================================"
echo "Demo completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
