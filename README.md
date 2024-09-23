# YTUnCoverLLM
An approach to process YouTube metadata to detect songs that are likely contained in videos.

# Getting Started

In the directory `baseline` the following must be included as a submodule: [baseline code](https://github.com/deezer/music-ner-eacl2023). 

## Baseline: music-ner-eacl2023 from Deezer researchers

To run the baseline experiments in the authors paper, run:

```sh
prepare_baseline_data.sh;
finetune_baseline.sh
```

## LLMs on music-ner-eacl2023 data
Specify the `llm`, `k` and `sampling`, for instance for `mixtral`, `k=25` and `tfidf`-sampling:
```
run_ie_pydantic.sh mixtral 25 tfidf
```

## Run experiments on our data

### Data preparation

This transforms DaTacos and SHS100K2 datasets into NER datasets. Entities from the SHS metadata are marked in the YouTube metadata.

```sh
prepare_csi_data.sh
```
TBA
