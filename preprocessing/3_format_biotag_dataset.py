import argparse
import os
import pandas as pd
import numpy as np


def main():

    args = parse_args()

    data = pd.read_parquet(args.input)

    data.TEXT = data.TEXT.apply(lambda x: np.append(x, None))
    data.NER_TAGS = data.NER_TAGS.apply(lambda x: np.append(x, None))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data.explode(["TEXT", "NER_TAGS"])[["TEXT", "NER_TAGS"]].to_csv(
        args.output, 
        sep="\t", 
        index=None, 
        header=False
        )

def parse_args():
    parser = argparse.ArgumentParser(description='Format parquet dataset with BIO tag lists to BIO format.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output BIO file.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()