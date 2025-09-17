#!/bin/bash
set -e
mkdir -p logs
declare -A experiments=(
    ["opt_deepspeed.py"]="deepspeed"
    ["opt_my.py"]="my" 
    ["opt_single.py"]="single"
)

echo "Начинаем запуск экспериментов..."
for script in "${!experiments[@]}"; do
    name="${experiments[$script]}"
    log_file="logs/${name}.log"
    
    echo "=================================================="
    echo "Запускаю: $script"
    echo "Лог файл: $log_file"
    echo "=================================================="
    
    if deepspeed --num_gpus=2 "$script" &> "$log_file"; then
        echo "✅ Эксперимент $script завершен успешно"
    else
        echo "❌ Эксперимент $script завершен с ошибкой"
        exit 1
    fi
    
    echo ""
done

echo "Все эксперименты завершены!"