#!/bin/bash
# ============================================================================
# SCAFFOLD Zero-Shot Training and Visualization Commands
# ============================================================================

echo ""
echo "========================================================================"
echo "SCAFFOLD ZERO-SHOT NOVELTY DETECTION - QUICK COMMANDS"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="D:/Final yr project/FRCOGRAPH/fecograph 1.0.2/fecograph"
VENV_PYTHON="D:/Final yr project/FRCOGRAPH/feco_env/Scripts/python.exe"

show_menu() {
    echo "Select an option:"
    echo ""
    echo "[1] Run Training (SCAFFOLD Zero-Shot)"
    echo "[2] Generate Visualizations (After Training)"
    echo "[3] Run Both (Training + Visualizations)"
    echo "[4] View Results Directory"
    echo "[5] Exit"
    echo ""
    read -p "Enter choice (1-5): " choice
    
    case $choice in
        1) run_training ;;
        2) run_visualization ;;
        3) run_both ;;
        4) view_results ;;
        5) exit 0 ;;
        *) echo "Invalid choice. Please try again."; show_menu ;;
    esac
}

run_training() {
    echo ""
    echo "========================================================================"
    echo -e "${BLUE}RUNNING TRAINING...${NC}"
    echo "========================================================================"
    echo ""
    
    cd "$PROJECT_DIR"
    "$VENV_PYTHON" run_scaffold_zeroshot.py
    
    echo ""
    echo -e "${GREEN}Training complete! Results saved to: results/scaffold_zeroshot/${NC}"
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

run_visualization() {
    echo ""
    echo "========================================================================"
    echo -e "${BLUE}GENERATING VISUALIZATIONS...${NC}"
    echo "========================================================================"
    echo ""
    
    cd "$PROJECT_DIR/visualizations"
    "$VENV_PYTHON" plot_results.py
    
    echo ""
    echo -e "${GREEN}Visualizations saved to: visualizations/output/${NC}"
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

run_both() {
    echo ""
    echo "========================================================================"
    echo -e "${BLUE}RUNNING TRAINING + VISUALIZATIONS...${NC}"
    echo "========================================================================"
    echo ""
    
    cd "$PROJECT_DIR"
    echo -e "${YELLOW}[1/2] Running training...${NC}"
    "$VENV_PYTHON" run_scaffold_zeroshot.py
    
    echo ""
    echo -e "${YELLOW}[2/2] Generating visualizations...${NC}"
    cd visualizations
    "$VENV_PYTHON" plot_results.py
    
    echo ""
    echo -e "${GREEN}Complete!${NC}"
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

view_results() {
    echo ""
    echo "Results locations:"
    echo "  - Training results: $PROJECT_DIR/results/scaffold_zeroshot/"
    echo "  - Visualizations:   $PROJECT_DIR/visualizations/output/"
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Start
show_menu
