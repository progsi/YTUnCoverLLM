# datacos
datacos_path="data/raw/datacos_yt.parquet"

echo "Creating datacos NER biotag dataset"
python preprocessing/1_preprocess_dataset.py -i "$datacos_path" -o data/intermediate/datacos_processed.parquet --split
python preprocessing/2_make_IOB_dataset.py -i data/intermediate/datacos_processed.parquet -o data/intermediate/datacos_IOB.parquet --all --baseline_names
python preprocessing/3_write_IOB_dataset.py -i data/intermediate/datacos_IOB.parquet -o data/intermediate/datacos/ --limit 3000
