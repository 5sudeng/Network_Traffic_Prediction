#!/bin/bash
config_dir="./configs/"
config_files=("lstm.json" "cnnlstm.json" "rnn.json" "gru.json")
# 에폭 (epochs) 설정
epochs=(20 40 60 80 100)

# Weight Decay 설정
weight_decays=(0 1e-4)

# step size 설정
step_sizes=(5 10)

# gamma 설정
gammas=(0.5 0.1)

for config_file in "${config_files[@]}"; do
    config_path="${config_dir}${config_file}"
    for epoch in "${epochs[@]}"; do
        for weight_decay in "${weight_decays[@]}"; do
            for step_size in "${step_sizes[@]}"; do
                for gamma in "${gammas[@]}"; do
                    # 실험 실행
                    python train.py -c "$config_path" --epoch "$epoch" --weight_decay "$weight_decay" --step_size "$step_size" --gamma "$gamma"
                done
            done
        done
    done
done