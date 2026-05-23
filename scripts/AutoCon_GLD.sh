#!/bin/bash

for wnorm in ReVIN Mean LastVal Decomp; do
    # 循环执行
    echo "========================================="
    echo "Weight Normalization: $wnorm"
    echo "========================================="
    for pred_len in {1..500..1}; do
        echo "[RUN] pred_len=${pred_len}"
        root_path="./dataset"
        data_path="GLD.csv"
        data="custom"
        target="Close"
            for seq_len in 16 32 64 96 128 192 256 320 384; do
            echo "-----------------------------------------"
            echo "Sequence length: $seq_len"
            echo "-----------------------------------------"
            label_len=$(echo $seq_len | awk '{print int($1/2)}')
            feature="S"
            model_name="AutoConNet"
            log_file="./logs/${model_name}_${wnorm}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
            python -u run_autocon.py \
            --AutoCon  \
            --AutoCon_multiscales 48 \
            --AutoCon_wnorm ${wnorm}  \
            --AutoCon_lambda 1.0   \
            --d_model 16 \
            --d_ff 16 \
            --e_layers 1 \
            --target Close \
            --c_out 1 \
            --root_path ./dataset/ \
            --data_path GLD.csv \
            --model_id ${model_name}_${wnorm}_GLD \
            --model $model_name \
            --data custom \
            --features $feature \
            --label_len $label_len \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 1 \
            --des 'Exp' \
            --batch_size 64 \
            --learning_rate 0.001 \
            --feature S > $log_file 2>&1
        done
    done
done