python convert_to_conv_data.py \
--orig_data ../../../openai-quickstart-python/dataset/5.1.2/finetune/5_1_2_finetune_raw_data.json \
--write_data 512raw_conv.json \
--dataset_name 512_ \
head -n 1000 512raw_conv.json > 512raw-dev1K.json
tail -n +1001 512raw_conv.json > 512raw.json