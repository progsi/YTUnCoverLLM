cd ..
shs100k2_path="data/raw/shs100k2_yt.parquet"

python preprocessing/4_make_dataset_parquet.py -i data/intermediate/shs100k2_IOB.parquet --human_annotation_file data/intermediate/shs100k2_annotated.parquet -o data/dataset/shs100k2/
