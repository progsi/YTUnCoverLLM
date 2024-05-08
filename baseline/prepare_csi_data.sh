cd ..
datacos_path="/data/csi_datasets/datacos_yt.parquet"
shs100k2_path="/data/csi_datasets/shs100k2_yt.parquet"

# datacos
echo "Creating datacos NER biotag dataset"
python preprocessing/1_preprocess_dataset.py -i "$datacos_path" -o data/datacos_processed.parquet
python preprocessing/2_make_biotag_dataset.py -i data/datacos_processed.parquet -o data/datacos_biotag.parquet --all --baseline_names
python preprocessing/3_format_biotag_dataset.py -i data/datacos_biotag.parquet -o baseline/music-ner-eacl2023/data/datacos/test.bio

# shs100k2
echo "Creating shs100k2 NER biotag dataset"
python preprocessing/1_preprocess_dataset.py -i "$shs100k2_path" -o data/shs100k2_processed.parquet
python preprocessing/2_make_biotag_dataset.py -i data/shs100k2_processed.parquet -o data/shs100k2_biotag.parquet --all --baseline_names
python preprocessing/3_format_biotag_dataset.py -i data/shs100k2_biotag.parquet -o baseline/music-ner-eacl2023/data/shs100k2/test.bio
