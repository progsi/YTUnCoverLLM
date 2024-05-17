import argparse
import pandas as pd
from src.Wrapper import NER_Wrapper


def main():

    args = parse_args()

    data = pd.read_parquet(args.input).set_index("yt_id")

    model = NER_Wrapper(args.model)

    # get rid of newline tags
    data[args.text_attribute] = data[args.text_attribute].str.replace("\n", " ")

    # make NER
    data = model.concat_entities(data, text_attrs=args.text_attribute, extract_attrs=args.extract_attributes)

    data.to_parquet(args.output)


def parse_args():
    parser = argparse.ArgumentParser(description='Run named entity recognition for song attribute extraction on an input dataset.')
    parser.add_argument('-i', '--input', type=str, help='Path with input parquet file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output parquet file.')
    parser.add_argument('-t','--text_attribute', type=str, help='<Required> Attribute to extract from', default="yt_processed")
    parser.add_argument('-e','--extract_attributes', nargs='+', help='<Required> Attributes to be extracted', default="title")
    parser.add_argument('-m', '--model', type=str, help='Model to use.', choices=["bert-large-uncased", "mpnet-base", "roberta-large"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()