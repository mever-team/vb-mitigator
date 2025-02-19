import os
import pandas as pd
import argparse

def annotate_tags(dataset_path):
    tags_file = os.path.join(dataset_path, "unique_tags.csv")
    greek_tags_file = os.path.join(dataset_path, "unique_tags_greek.csv")

    output_file = os.path.join(dataset_path, "manual_annotations.csv")
    
    if not os.path.exists(tags_file):
        print(f"Skipping {dataset_path}: unique_tags.csv not found.")
        return
    if not os.path.exists(greek_tags_file):
        print(f"Skipping {dataset_path}: unique_tags_greek.csv not found.")
        return
    
    df = pd.read_csv(tags_file)
    if "Tags" not in df.columns:
        print(f"Skipping {dataset_path}: 'Tags' column not found in unique_tags.csv.")
        return
    

    greek_df = pd.read_csv(greek_tags_file)
    if "Tags" not in greek_df.columns:
        print(f"Skipping {dataset_path}: 'Tags' column not found in tags_greek.csv.")
        return
    annotations = []
    print("Enter class or 'a' if related to all classes (press enter for 'n')")
    for tag, greek_tag in zip(df["Tags"],greek_df["Tags"]):
        
        while True:
            user_input = input(f"Tag: {tag} ({greek_tag}): ").strip()
            if user_input.isdigit() or user_input.lower() == "a" or user_input == "":
                annotations.append((tag, user_input if user_input else "n"))
                break
            else:
                print("Invalid input. Please enter a digit, 'a', or press enter for 'n'.")
    
    annotated_df = pd.DataFrame(annotations, columns=["tags", "annotations"])
    annotated_df.to_csv(output_file, index=False)
    print(f"Annotations saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of the dataset to process")
    args = parser.parse_args()
    
    dataset_path = os.path.join("./data", args.dataset)
    if os.path.isdir(dataset_path):
        print(f"Processing dataset: {args.dataset}")
        annotate_tags(dataset_path)
    else:
        print(f"Dataset {args.dataset} not found.")

if __name__ == "__main__":
    main()