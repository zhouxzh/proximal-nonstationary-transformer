#!/bin/bash
set -e

mkdir -p ./logs

# Loop through prediction lengths from 1 to 500
for pred_len in {135..500..5}
do
  echo "========================================="
  echo "Running for prediction length: $pred_len"
  echo "========================================="

  # Common parameters
  # Note: Some models might override these
  root_path="./dataset"
  data_path="GLD.csv"
  data="custom"
  target="Close"
  for seq_len in 16 32 64 96 128 192 256 320 384; do
    echo "-----------------------------------------"
    echo "Sequence length: $seq_len"
    echo "Current Time: $(date)"
    echo "-----------------------------------------"
    label_len=$(echo $seq_len | awk '{print int($1/2)}')
    feature="S"
    
    # -------------------------------------------------------------------------
    # 1. Autoformer
    # -------------------------------------------------------------------------
    # model_name="Autoformer"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --target $target \
    #   --is_training 1 \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --des 'Exp' \
    #   --itr 1 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 2. Crossformer
    # -------------------------------------------------------------------------
    # model_name="Crossformer"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --target $target \
    #   --is_training 1 \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --d_model 64 \
    #   --d_ff 64 \
    #   --top_k 5 \
    #   --des 'Exp' \
    #   --itr 1 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 3. FiLM
    # -------------------------------------------------------------------------
    # model_name="FiLM"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --target $target \
    #   --is_training 1 \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --des 'Exp' \
    #   --itr 1 > $log_file 2>&1
      
    # -------------------------------------------------------------------------
    # 4. MICN
    # -------------------------------------------------------------------------
    # model_name="MICN"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --target $target \
    #   --is_training 1 \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --d_model 64 \
    #   --d_ff 64 \
    #   --top_k 5 \
    #   --des 'Exp' \
    #   --itr 1 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 5. Nonstationary_Transformer
    # -------------------------------------------------------------------------
    # model_name="Nonstationary_Transformer"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --target $target \
    #   --is_training 1 \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --d_model 64 \
    #   --d_ff 64 \
    #   --top_k 5 \
    #   --des 'Exp' \
    #   --itr 1 \
    #   --p_hidden_dims 256 256 \
    #   --p_hidden_layers 2 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 6. PatchTST
    # -------------------------------------------------------------------------
    # model_name="PatchTST"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --is_training 1 \
    #   --target $target \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --des 'Exp' \
    #   --itr 1 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 7. Pyraformer
    # -------------------------------------------------------------------------
    model_name="Pyraformer"
    log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    echo "Running $model_name..."
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --target $target \
      --root_path $root_path \
      --data_path $data_path \
      --model_id ${model_name}_GLD \
      --model $model_name \
      --data $data \
      --features ${feature} \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 8. TimesNet
    # -------------------------------------------------------------------------
    # model_name="TimesNet"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --is_training 1 \
    #   --target $target \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --d_model 64 \
    #   --d_ff 64 \
    #   --top_k 5 \
    #   --des 'Exp' \
    #   --itr 1 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 9. Transformer
    # -------------------------------------------------------------------------
    # model_name="Transformer"
    # log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    # echo "Running $model_name..."
    # python -u run.py \
    #   --task_name long_term_forecast \
    #   --is_training 1 \
    #   --target $target \
    #   --root_path $root_path \
    #   --data_path $data_path \
    #   --model_id ${model_name}_GLD \
    #   --model $model_name \
    #   --data $data \
    #   --features ${feature} \
    #   --seq_len $seq_len \
    #   --label_len $label_len \
    #   --pred_len $pred_len \
    #   --e_layers 2 \
    #   --d_layers 1 \
    #   --factor 3 \
    #   --enc_in 1 \
    #   --dec_in 1 \
    #   --c_out 1 \
    #   --des 'Exp' \
    #   --itr 1 > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 10. NLinear
    # -------------------------------------------------------------------------
    model_name="NLinear"
    log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    echo "Running $model_name..."
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path GLD.csv \
      --model_id ${model_name}_GLD \
      --model $model_name \
      --data custom \
      --features $feature \
      --label_len $label_len \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 1 \
      --enc_in 1 \
      --des 'Exp' \
      --target $target \
      --itr 1 --batch_size 64  --learning_rate 0.005 --individual > $log_file 2>&1

    # -------------------------------------------------------------------------
    # 11. DLinear
    # -------------------------------------------------------------------------
    model_name="DLinear"
    log_file="./logs/${model_name}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    echo "Running $model_name..."
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path GLD.csv \
      --model_id ${model_name}_GLD \
      --model $model_name \
      --data custom \
      --features $feature \
      --label_len $label_len \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 1 \
      --enc_in 1 \
      --des 'Exp' \
      --target $target \
      --itr 1 --batch_size 64 --learning_rate 0.0005 --individual > $log_file 2>&1
      
    # -------------------------------------------------------------------------
    # 12. Proximal_Nonstationary_Transformer (with loop over wnorm)
    # -------------------------------------------------------------------------
    # model_name="Proximal_Nonstationary_Transformer"
    # for wnorm in LastVal Mean Median; do
    #     log_file="./logs/${model_name}_${wnorm}_GLD_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log"
    #     echo "Running $model_name with wnorm=$wnorm..."
    #     python -u run.py \
    #     --task_name long_term_forecast \
    #     --target $target \
    #     --is_training 1 \
    #     --root_path ./dataset/ \
    #     --data_path GLD.csv \
    #     --model_id ${model_name}_${wnorm}_GLD \
    #     --model $model_name \
    #     --wnorm ${wnorm} \
    #     --data custom \
    #     --features $feature \
    #     --label_len $label_len \
    #     --seq_len $seq_len \
    #     --pred_len $pred_len \
    #     --e_layers 2 \
    #     --d_layers 1 \
    #     --factor 3 \
    #     --enc_in 1 \
    #     --dec_in 1 \
    #     --c_out 1 \
    #     --d_model 64 \
    #     --d_ff 64 \
    #     --top_k 5 \
    #     --des 'Exp' \
    #     --itr 1 \
    #     --p_hidden_dims 8 8 \
    #     --p_hidden_layers 2 > $log_file 2>&1
    # done
  done
done
