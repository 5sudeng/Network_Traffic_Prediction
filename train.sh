config_dir="./configs/"

config_files=("lstm.json" "cnnlstm.json" "rnn.json" "gru.json")

for config_file in "${config_files[@]}"; do
    config_path="${config_dir}${config_file}"
    python train.py -c "$config_path" 
done