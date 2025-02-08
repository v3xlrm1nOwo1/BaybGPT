python train.py --config_path "config.ini" \
               --model_config_section "Model" \
               --train_batch_size 32 \
               --eval_batch_size 64 \
               --epoch 100 \
               --learning_rate 6e-4 \
               --seed 1234 \
               --num_generated_tokens 512\
               --show_generated_text True