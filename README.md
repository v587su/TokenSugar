# Token Sugar
This repository contains the code for the paper "Token Sugar: Making Source Code Sweeter for LLMs through Token-Efficient Shorthand"

# Data

The token sugars mined in this paper are stored in the `mined_sugars.json` file. You can revise `python miner/mine.py` to mine yours.


# Mining

To mine the token sugars from `"LimYeri/LeetCode_Python_Solutions_v2`, run the following command:

```bash
python miner/mine.py --threshold [THRESHOLD_FOR_APPEARANCE] --min_reward [MINIMUM_TOKEN_TO_BE_SAVED] --use_pool
```

# Training
Before training, you may need to set the HuggingFace token in the environment variable `HUGGINGFACE_TOKEN`.

The training scripts are in the file `train.py`. It is a revised version of [MagiCoder](https://github.com/ise-uiuc/magicoder).

The command to train a model is as follows:



```bash
accelerate launch -m train \
  --model_key $MODEL_KEY \
  --use_flash_attention True \
  --max_training_seq_length 512 \
  --datafile_paths $PATH_TO_DATASET_FILE \
  --output_dir $OUTPUT_DIR \
  --bf16 True \
  --num_train_epochs EPOCHS \
  --per_device_train_batch_size BATCH_SIZE \
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS \
  --group_by_length False \
  --ddp_find_unused_parameters False \
  --logging_steps 50 \
  --log_level info \
  --warmup_steps WARMUP_STEPS \
  --learning_rate LEARNING_RATE \
  --lr_scheduler_type cosine \
  --load_from new \
  --is_sugarized False  \
  --pattern_file_path PATH_TO_SUGAR_FILE \
  --eval_dataset_size EVAL_DATASET_SIZE \
  --eval_steps EVAL_STEPS \
  --evaluation_strategy steps \
  --save_strategy epoch \
  --task pretrain \
  --source_data starcoder \
  --num_proc NUM_PROC \
  --start_id START_ID  # if this is set to 100, the special tokens will be mapped to the ids from 100 to 100 + the number of special tokens
```

We use bigcode-evaluation-harness to evaluate the models. The command to evaluate a model is as follows:

```bash
cd bigcode-evaluation-harness

accelerate launch main.py \
  --model $PATH_TO_MODEL  \
  --tasks humaneval \
  --max_length_generation 512 \
  --temperature 0.0 \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --pattern_filename PATH_TO_SUGAR_FILE \
  --save_generations_path PATH_TO_SAVE_GENERATIONS \
  --metric_output_path PATH_TO_SAVE_METRIC \
  --start_id START_ID 
```
