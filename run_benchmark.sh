#!/bin/bash

model=$1  
dataset=$2  

# Pass the string with values separated by commas
input_string=$3

# Set IFS to a comma to split the string into individual elements
IFS=',' read -r -a array <<< "$input_string"


# Get the number of datasets (assuming directories are named dataset1, dataset2, etc.)
d_id_list=$(ls -d "data/dataset/"$dataset"/dataset"*"/" | grep -o 'dataset[0-9]*' | grep -o '[0-9]*')

# Loop through each dataset ID
for d_id in $d_id_list; do

    # Loop through each value of k
    for k in "${array[@]}"; do

        # If-Else condition
        if [ "$k" -eq 0 ]; then
            python run_ie_json.py --llm "$model" -i "data/dataset/${dataset}/dataset${d_id}/test.IOB" -k 0 -o "output/${dataset}/${model}/dataset${d_id}_0shot.jsonl"
        else
            for sampling in rand tfidf; do
                python run_ie_json.py --llm $model -i "data/dataset/"$dataset"/dataset"$d_id"/test.IOB" -k $k -s $sampling -o "output/"$dataset"/"$model"/dataset"$d_id"_"$k"shot_"$sampling".jsonl" 
            done
        fi  
    done
done
