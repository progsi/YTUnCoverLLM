# Activate conda env and go to dir
#!/bin/bash

DATASET_NAME=$1

# Initialize Conda (if not already initialized)
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if Conda is initialized
if ! [[ $PATH =~ "/data/miniconda3/bin/activate" ]]; then
    echo "Initializing Conda..."
    source /data/miniconda3/bin/activate
fi

# 2. Define the name of the Conda environment
conda_env="music-ner-eacl2023"

# 3. Check if the Conda environment exists
if ! conda env list | grep -q "^$conda_env "; then
    echo "Conda environment '$conda_env' not found. Creating it..."
    conda create --name "$conda_env" python=3.8
fi

# 4. Activate the Conda environment
echo "Activating Conda environment '$conda_env'..."
conda activate "$conda_env"


cd music-ner-eacl2023

MODEL="roberta-large"
BATCH_SIZE=16
NUM_EPOCHS=3
SAVE_STEPS=750
REINIT_LAYERS=1
SEED=1
for DS_ID in 1 2 3 4
do
	DATA_DIR="data/"$DATASET_NAME"/dataset"$DS_ID
    mkdir DATA_DIR="data/"$DATASET_NAME
    mkdir $DATA_DIR
    
    BASE_NAME=$(basename ${MODEL})
    OUTPUT_DIR="output/"$DATASET_NAME"/dataset"$DS_ID

    cp "../../data/dataset/"$DATASET_NAME"/dataset"$DS_ID"/test.IOB" $DATA_DIR"/test.bio"
    cp "../../data/dataset/"$DATASET_NAME"/dataset"$DS_ID"/train.IOB" $DATA_DIR"/train.bio"
    python music-ner/src/fine-tune.py --dataset_name music-ner/datasets --model_name_or_path $MODEL --output_dir $OUTPUT_DIR --num_train_epochs $NUM_EPOCHS --per_device_train_batch_size $BATCH_SIZE --seed $SEED --do_train --do_predict --overwrite_output_dir  --reinit_layers $REINIT_LAYERS --return_entity_level_metrics --dataset_path=$DATA_DIR
done