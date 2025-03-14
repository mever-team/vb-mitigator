import pandas as pd
import ast
import argparse

def extract_unique_tags(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    unique_tags = set()
    
    for tag_list in df["tags"]:
        tags = ast.literal_eval(tag_list)  # Convert string representation of list to actual list
        unique_tags.update(tags)
    
    unique_df = pd.DataFrame(sorted(unique_tags), columns=["Tags"])
    unique_df.to_csv(output_csv, index=False)
    print(f"Unique tags saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="Path to input CSV file")
    parser.add_argument("--output_csv", help="Path to output CSV file")
    args = parser.parse_args()
    
    extract_unique_tags(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
