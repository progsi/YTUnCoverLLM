cd ..
shs100k2_path="data/raw/shs100k2_yt.parquet"

python preprocessing/4_finalize_dataset.py -i data/intermediate/shs100k2_IOB.parquet -a data/annotated/ -o data/dataset/shsyt/
