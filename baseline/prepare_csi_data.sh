cd ..
datacos_path="data/raw/datacos_yt.parquet"
shs100k2_path="data/raw/shs100k2_yt.parquet"

# shs100k2
echo "Creating shs100k2 NER biotag dataset"
python preprocessing/1_preprocess_dataset.py -i "$shs100k2_path" -o data/intermediate/shs100k2_processed.parquet --split
python preprocessing/2_make_biotag_dataset.py -i data/intermediate/shs100k2_processed.parquet -o data/intermediate/shs100k2_biotag.parquet --all --baseline_names
python preprocessing/3_format_biotag_dataset.py -i data/intermediate/shs100k2_biotag.parquet -o baseline/music-ner-eacl2023/data/shs100k2/ --minimum_ents 2

# datacos
echo "Creating datacos NER biotag dataset"
python preprocessing/1_preprocess_dataset.py -i "$datacos_path" -o data/intermediate/datacos_processed.parquet --split
python preprocessing/2_make_biotag_dataset.py -i data/intermediate/datacos_processed.parquet -o data/intermediate/datacos_biotag.parquet --all --baseline_names
python preprocessing/3_format_biotag_dataset.py -i data/intermediate/datacos_biotag.parquet -o baseline/music-ner-eacl2023/data/datacos/ --minimum_ents 2
cp baseline/music-ner-eacl2023/data/shs100k2/train.bio  baseline/music-ner-eacl2023/data/datacos/