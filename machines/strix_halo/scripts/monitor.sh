#!/bin/bash
# GPU/Memory monitor for Strix Halo training runs
# Usage: bash monitor.sh [interval_seconds]
INTERVAL=${1:-10}
LOG="machines/strix_halo/results/monitor_$(date +%Y%m%d_%H%M%S).log"

echo "Monitoring GPU/RAM every ${INTERVAL}s → $LOG"
echo "timestamp,gpu_temp,gpu_power,vram_pct,gpu_pct,ram_used_gb,ram_free_gb" > "$LOG"

while true; do
    TS=$(date +%H:%M:%S)
    GPU=$(rocm-smi 2>&1 | grep "^0" | awk '{print $5","$6","$14","$15}' | tr -d '%°CW')
    RAM=$(free -g | awk '/^Mem:/{print $3","$4}')
    echo "$TS,$GPU,$RAM" >> "$LOG"
    echo "$TS  GPU: $GPU  RAM: ${RAM}GB"
    sleep $INTERVAL
done
