#!/bin/bash
#
# Production Run Script for NLAA Analyzer
# Processes all ~1M records with checkpointing and logging
#
# Usage:
#   ./run_production.sh          # Start fresh or auto-resume
#   ./run_production.sh --fresh  # Clear checkpoint and start fresh
#

set -e

cd "$(dirname "$0")"

# Configuration
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/production_run_${TIMESTAMP}.log"
CHECKPOINT_INTERVAL=100
RATE_LIMIT=2.0

# Create log directory
mkdir -p "$LOG_DIR"

# Check for --fresh flag
if [[ "$1" == "--fresh" ]]; then
    echo "Clearing existing checkpoint..."
    python nlaa_analyzer.py --clear-checkpoint
    RESUME_FLAG=""
else
    # Auto-resume if checkpoint exists
    if [[ -f "checkpoints/checkpoint.json" ]]; then
        echo "Found existing checkpoint - will resume"
        RESUME_FLAG="--resume"
    else
        echo "No checkpoint found - starting fresh"
        RESUME_FLAG=""
    fi
fi

echo ""
echo "=================================================="
echo "  NLAA Analyzer - Production Run"
echo "=================================================="
echo ""
echo "  Records:     ~1,075,391"
echo "  Rate limit:  ${RATE_LIMIT} req/sec"
echo "  Est. time:   ~6 days"
echo "  Log file:    ${LOG_FILE}"
echo "  Checkpoint:  every ${CHECKPOINT_INTERVAL} records"
echo ""
echo "  Press Ctrl+C to stop (progress will be saved)"
echo ""
echo "=================================================="
echo ""

# Run the analyzer
python nlaa_analyzer.py \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --rate-limit "$RATE_LIMIT" \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Run complete! Log saved to: ${LOG_FILE}"

