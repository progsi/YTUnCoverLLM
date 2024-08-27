import pandas as pd

data = pd.read_parquet("../data/annotated/data_annotated.parquet").reset_index()

# filter for samples with two annotations
col_annot1 = ("IOB", "ANNOT1")
col_annot2 = ("IOB", "ANNOT2")
col_annot5 = ("IOB", "ANNOT5")

mask = ~data[col_annot5].isna() & (~data[col_annot1].isna() | ~data[col_annot2].isna())
data = data[mask]
data

_da1 = pd.read_csv("../data/annotated/disagreement_ANNOT1_ANNOT5_curated.csv", sep="\t").rename(
    columns={"ANNOT1": "a1", "ANNOT5": "a2"})
_da2 = pd.read_csv("../data/annotated/disagreement_ANNOT2_ANNOT5_curated.csv", sep="\t").rename(
    columns={"ANNOT2": "a1", "ANNOT5": "a2"})
data_curated = pd.concat([_da1, _da2])

def get_correct(row):
    x = row.Curated
    if x == 1 or x == "1":
        return row.a1
    elif x == 2 or x == "2":
        return row.a2
    else:
        return row.Curated

data_curated.Curated = data_curated.apply(get_correct, axis=1)

def get_final(row):
    set_id, yt_id = row.set_id.item(), row.yt_id.item()
    mask = (data_curated.set_id == set_id) & (data_curated.yt_id == yt_id)
    curated = data_curated.loc[mask, "Curated"]
    if not curated.empty:
        iob = curated.item()
    else:
        iob = row[("IOB", "ANNOT5")]
    return iob

data[("IOB", "final")] = data.apply(get_final, axis=1)
data.to_json("../data/dataset/shs100k2/data_annotated.jsonl", orient="records", lines=True)


