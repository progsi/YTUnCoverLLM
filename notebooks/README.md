# Notebooks
- `dataset.ipynb`: dataset analysis
- `matching_pairwise.ipynb`: matching baseline of video metadata with their corresponding ground truth attributes. To gather insights about how often attributes are in the metadata, etc.
- `matching_pairwise_ner.ipynb`: like the first one but looking at NER extracted attributes. 
- `matching_cartesian.ipynb`: matching baseline for all pairs of items. Here we evaluate with mean average precision score like in traditional cover song identification.
- `IOB_analysis.ipynb`: to analyze the effects of preprocessing on the parquet files with IOB tags in the `data` subdir. 
- `prepare_annotation.ipynb`: to prepare data for human annotation 
- `llm_playground.ipynb`: playing around with Llama-3
- `performer_strings.ipynb`: Analysis about performer strings and how to split them into multiple performers
- `baseline_results.ipynb`: Analysis of results of the baseline models (BERT, RoBERTa, MPNet)
- `wikidata.ipynb`: Processing of the wikidata crawl and creation of the joint dataset (Reddit+SHS-YT)