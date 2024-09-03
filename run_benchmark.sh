#!/bin/bash

model=$1  
dataset=$2  

# Get the number of datasets (assuming directories are named dataset1, dataset2, etc.)
d_id_list=$(ls -d "data/dataset/"$dataset"/dataset"*"/" | grep -o 'dataset[0-9]*' | grep -o '[0-9]*')

for sampling in rand tfidf; do
    # Loop through each dataset ID
    for d_id in $d_id_list; do
        # Loop through each value of k
        for k in 5 15 25 35; do
            # Construct and execute the command
            python run_ie_json.py --llm $model -i "data/dataset/"$dataset"/dataset"$d_id"/test.IOB" -k $k -s $sampling -o "output/"$dataset"/"$model"/dataset"$d_id"_"$k"shot_"$sampling".jsonl" 
        done
    done
done
