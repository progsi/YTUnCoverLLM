import argparse
import csv
import sys
import os
sys.path.append(os.path.abspath("."))
from preprocessing.Utils import simplify_string
from src.Utils import read_IOB_file
from preprocessing.Utils import unicode_normalize

def write_iob_file(words_list, tags_list, output_path):
    with open(output_path, 'w') as f:
        for words, tags in zip(words_list, tags_list):
            for word, tag in zip(words, tags):
                f.write(f"{word}\t{tag}\n")
            f.write("\n")

def align_tags(original_text, processed_text, processed_tags):
    original_tags = []

    original_text = [unicode_normalize(s) for s in original_text]
    processed_text = [unicode_normalize(s) for s in processed_text]

    cls_order = [s for s in processed_tags if s.startswith("B-")]

    assert len(processed_text) == len(processed_tags), "Tags do not match text!"
    i, j = 0, 0 # i for processed and j for unprocessed

    while j < len(original_text):
        # normal copying
        if i < len(processed_text) and processed_text[i] == original_text[j]:
            original_tags.append(processed_tags[i])
            i += 1
        # special chars
        elif i < len(processed_text) and processed_text[i] == '|':
            original_tags.append(processed_tags[i])
            i += 1
        else:
            if (j > 0 and j > 0 and 
                original_tags[-1].startswith("I-") and 
                (i < len(processed_tags) and processed_tags[i].startswith("I-")) and 
                processed_tags[i] == processed_tags[i-1]):
                original_tags.append(original_tags[-1])
            else:
                original_tags.append("O")
        j += 1

    while j < len(original_text):
        original_tags.append("O")
        j += 1

    assert len(original_tags) == len(original_tags), "Aligned tags do not match text!"
    if not cls_order == [s for s in original_tags if s.startswith("B-")]:
        return None
    return original_tags

def process_files(csv_file, iob_file):

    original_texts = []
    original_tags = []

    processed_texts, tags = read_IOB_file(iob_file)
    
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            original_text = simplify_string(row['original'].lower())
            
            aligned_tags = align_tags(original_text.split(), processed_texts[i], tags[i])
            if aligned_tags is None:
                continue

            print(f"Original: {original_text}")
            print(f"Tagged: {aligned_tags}")

            original_texts.append(original_text.split())
            original_tags.append(aligned_tags)
    
    return original_texts, original_tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align IOB tags from preprocessed text to the original text.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file with 'preprocessed' and 'original' columns.")
    parser.add_argument('--iob_file', type=str, required=True, help="Path to the IOB file corresponding to the preprocessed text.")
    parser.add_argument('--output', type=str, required=True, help="Path for new IOB file.")

    args = parser.parse_args()
    
    # Process the files and align tags
    words, tags = process_files(args.csv_file, args.iob_file)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_iob_file(words, tags, args.output)