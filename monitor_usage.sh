#!/bin/bash

# Запустити скрипт у фоновому режимі
python3 test.py &
PID=$!

# Збирати дані про GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1 > results/gpu_usage.csv &

# Збирати дані про CPU
pidstat -p $PID 1 > results/cpu_usage.txt &

# Чекати завершення скрипта
wait $PID

# Зупинити моніторинг
killall nvidia-smi pidstat

echo "Дані збережено у gpu_usage.csv та cpu_usage.txt"
