import pandas as pd
import argparse
from llamaindex import GPTLLM  # Replace this with the correct import path for your LLM

def query_llm_about_song(llm, title, performer):
    """Query the LLM to find the performing artist and verify the performer."""
    response = {}

    # Ask the LLM who the performing artist is for the given song title
    artist_query = f"Who is the performing artist of the song titled '{title}'?"
    artist_response = llm.query(artist_query)

    response['predicted_artist'] = artist_response.get('artist', 'Unknown')

    # Ask the LLM whether it knows the performer mentioned in the "performer" column
    performer_query = f"Do you know the performer '{performer}'?"
    performer_response = llm.query(performer_query)

    response['performer_known'] = performer_response.get('known', 'Unknown')

    return response

def main(llm_name, input_file, output_file):
    # Load the LLM
    llm = GPTLLM(llm_name)  # Replace this with actual LLM loading/initialization code

    # Read the input file
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, sep=';')
    elif input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")

    # Prepare the output DataFrame
    results = []

    for _, row in df.iterrows():
        title = row['title']
        performer = row['performer']
        response = query_llm_about_song(llm, title, performer)
        response['title'] = title
        response['performer'] = performer
        results.append(response)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query an LLM to find performing artists and verify performers in song data.")
    parser.add_argument("--llm", type=str, required=True, help="Name or path of the language model to use.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV or Parquet file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV file.")

    args = parser.parse_args()
    main(args.llm, args.input, args.output)
