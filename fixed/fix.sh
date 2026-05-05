#!/usr/bin/env bash

set -e

# список файлов без ".pdf"
names=(
  "opt-1.3b"
  "opt-1.3b-prefill"
  "opt-2.7b"
  "opt-6.7b"
  "opt-13b"
  "bloom-7b1"
  "deberta"
  "dinov2"
  "intern"
)

mkdir -p fixed

for name in "${names[@]}"; do
    input="${name}.pdf"
    output="fixed/${name}.pdf"

    if [[ ! -f "$input" ]]; then
        echo "[warn] файл '$input' не найден, пропускаю"
        continue
    fi

    echo "[info] Обрабатываю $input → $output"

    gs -sDEVICE=pdfwrite \
       -dCompatibilityLevel=1.4 \
       -dNOPAUSE -dQUIET -dBATCH \
       -sOutputFile="$output" \
       "$input"
done

echo "[done] Готово!"