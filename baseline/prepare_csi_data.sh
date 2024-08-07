cd ..
shs100k2_path="data/raw/shs100k2_yt.parquet"

# shs100k2
echo "Creating shs100k2 NER biotag dataset"
python preprocessing/1_preprocess_dataset.py -i "$shs100k2_path" -o data/intermediate/shs100k2_processed.parquet --split
python preprocessing/2_make_IOB_dataset.py -i data/intermediate/shs100k2_processed.parquet -o data/intermediate/shs100k2_IOB.parquet --all --baseline_names
python preprocessing/3_write_IOB_dataset.py -i data/intermediate/shs100k2_IOB.parquet -o data/intermediate/shs100k2/ --limit 3000
