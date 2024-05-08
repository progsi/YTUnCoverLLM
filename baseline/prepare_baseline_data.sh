cd music-ner-eacl2023
python music-ner/datasets/stats.py --data_dir=data/dataset1
python music-ner/datasets/stats.py --data_dir=data/dataset2
python music-ner/datasets/stats.py --data_dir=data/dataset3
python music-ner/datasets/stats.py --data_dir=data/dataset4

python music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset1/ --th_seen=1 --th_rare_unseen=0
python music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset2/ --th_seen=1 --th_rare_unseen=0
python music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset3/ --th_seen=1 --th_rare_unseen=0
python music-ner/datasets/create_seen_rare_ds.py --data_dir data/dataset4/ --th_seen=1 --th_rare_unseen=0
mkdir output