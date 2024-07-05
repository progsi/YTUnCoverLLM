#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <llm> <k>"
    exit 1
fi

# Assign input arguments to variables
llm=$1
k=$2

python run_ie_pydantic.py --llm "$llm" -k "$k" -i baseline/music-ner-eacl2023/data/dataset1/test.bio -o "output/"$llm"/dataset1_"$k"shot.jsonl" 
python run_ie_pydantic.py --llm "$llm" -k "$k" -i baseline/music-ner-eacl2023/data/dataset2/test.bio -o "output/"$llm"/dataset2_"$k"shot.jsonl"
python run_ie_pydantic.py --llm "$llm" -k "$k" -i baseline/music-ner-eacl2023/data/dataset3/test.bio -o "output/"$llm"/dataset3_"$k"shot.jsonl" 
python run_ie_pydantic.py --llm "$llm" -k "$k" -i baseline/music-ner-eacl2023/data/dataset4/test.bio -o "output/"$llm"/dataset4_"$k"shot.jsonl" 