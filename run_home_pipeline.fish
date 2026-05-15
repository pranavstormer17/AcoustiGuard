#!/usr/bin/env fish

# AcoustiGuard Pipeline Automation
# Executes Phase 1 (Baseline) and Phase 2 (Masked) sequentially.

mkdir -p logs
set TIMESTAMP (date +%Y%m%d_%H%M%S)
set LOGFILE "logs/pipeline_log_$TIMESTAMP.json"
set TEXTLOG "logs/pipeline_log_$TIMESTAMP.log"

echo "[INFO] AcoustiGuard Pipeline initiated at $(date)"
echo "[" > $LOGFILE
echo "Pipeline execution log started at $(date)" | tee $TEXTLOG

function run_cmd --argument-names phase command
    echo "[INFO] PHASE $phase: Executing $command" | tee -a $TEXTLOG
    set start (date +%s)
    
    set output (eval $command 2>&1 | tee -a $TEXTLOG)
    set rc $pipestatus[1]
    
    if test $rc -ne 0
        echo "[FATAL] $command terminated with exit code $rc" | tee -a $TEXTLOG
        
        # --- PERMANENT JSON FIX ---
        # Cleanly close the JSON file before aborting the script
        sed -i '$ s/,$//' $LOGFILE
        echo -e "\n]" >> $LOGFILE
        # --------------------------
        
        exit $rc
    end
    
    set end (date +%s)
    set duration (math $end - $start)
    
    set tail_output (echo "$output" | tail -n 15)
    set json_output (python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$tail_output")
    
    echo "  {" >> $LOGFILE
    echo "    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> $LOGFILE
    echo "    \"phase\": \"$phase\"," >> $LOGFILE
    echo "    \"command\": \"$command\"," >> $LOGFILE
    echo "    \"duration_seconds\": $duration," >> $LOGFILE
    echo "    \"output\": $json_output" >> $LOGFILE
    echo "  }," >> $LOGFILE
    
    echo "[SUCCESS] Task completed in $duration seconds\n" | tee -a $TEXTLOG
end

function set_config_mode
    set target_mode $argv[1]
    python3 -c "
import re, sys
with open('config.py', 'r') as f: content = f.read()
new_content = re.sub(r'^MODE\s*=\s*\"[^\"]+\"', 'MODE = \"' + sys.argv[1] + '\"', content, flags=re.M)
with open('config.py', 'w') as f: f.write(new_content)
" $target_mode | tee -a $TEXTLOG
end

echo "[INFO] Commencing Phase 1: Baseline Evaluation" | tee -a $TEXTLOG
set_config_mode "home"
run_cmd "1" "python -m src.preprocess_data"
run_cmd "1" "python sanity_check.py"
run_cmd "1" "python -m src.train_models"
run_cmd "1" "python -m src.evaluate_models"

echo "[INFO] Commencing Phase 2: Defense Evaluation" | tee -a $TEXTLOG
run_cmd "2" "python -m src.masker"
set_config_mode "home_masked"
run_cmd "2" "python -m src.preprocess_data"
run_cmd "2" "python sanity_check.py"
run_cmd "2" "python -m src.evaluate_models"

set_config_mode "home"
sed -i '$ s/,$//' $LOGFILE
echo -e "\n]" >> $LOGFILE

echo "[INFO] Pipeline execution complete. Metrics appended to JSON."