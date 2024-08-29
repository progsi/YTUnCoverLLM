import argparse
import csv
import sys
import os
sys.path.append(os.path.abspath("."))
from preprocessing.Utils import simplify_string
from src.Utils import read_IOB_file


def write_iob_file(words_list, tags_list, output_path):
    with open(output_path, 'w') as f:
        for words, tags in zip(words_list, tags_list):
            for word, tag in zip(words, tags):
                f.write(f"{word}\t{tag}\n")
            # Add an empty line to indicate separation between inner lists
            f.write("\n")

def align_tags(original_text, processed_text, processed_tags):
    # Initialize the output list of tags
    original_tags = []
    original_text = original_text.split()

    assert len(processed_text) == len(processed_tags), "Tags do not match text!"
    # Initialize indices for processed_text and original_text
    # Initialize indices for processed_text and original_text
    i, j = 0, 0

    # Iterate over both strings
    while j < len(original_text):
        if i < len(processed_text) and processed_text[i] == original_text[j]:
            # Directly copy the tag if characters match
            original_tags.append(processed_tags[i])
            i += 1
        elif i < len(processed_text) and processed_text[i] == '|':
            # Replace '|' in processed_text with the corresponding char in original_text
            original_tags.append(processed_tags[i])
            i += 1
        else:
            # Handle the additional characters in original_text
            if (j > 0 and j < len(original_text) - 1 and 
                original_tags[-1].startswith("I-") and 
                (i < len(processed_tags) and processed_tags[i].startswith("I-")) and 
                processed_tags[i] == processed_tags[i-1]):
                # If between two I- tags of the same class, assign the same I- tag
                original_tags.append(original_tags[-1])
            else:
                # Otherwise, assign an 'O' tag
                original_tags.append("O")
        j += 1

    # Handle the case where there are leftover characters in original_text
    while j < len(original_text):
        original_tags.append("O")
        j += 1

    assert len(original_tags) == len(original_tags), "Aligned tags do not match text!"

    return original_tags

def process_files(csv_file, iob_file):

    original_texts = []
    original_tags = []

    processed_texts, tags = read_IOB_file(iob_file)
    
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            original_text = simplify_string(row['original'].lower())
            
            aligned_tags = align_tags(original_text, processed_texts[i], tags[i])

            print(f"Original: {original_text}")
            print(f"Tagged: {aligned_tags}")

            original_texts.append(original_text.split())
            original_tags.append(aligned_tags)
    
    return original_texts, original_tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align IOB tags from preprocessed text to the original text.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file with 'preprocessed' and 'original' columns.")
    parser.add_argument('--iob_file', type=str, required=True, help="Path to the IOB file corresponding to the preprocessed text.")
    parser.add_argument('--output', type=str, required=True, help="Path to the IOB file corresponding to the preprocessed text.")

    args = parser.parse_args()
    
    # Process the files and align tags
    words, tags = process_files(args.csv_file, args.iob_file)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_iob_file(words, tags, args.output)