import os
import pandas as pd
import argparse

def compute_metrics(dataset_path):
    manual_annotations_file = os.path.join(dataset_path, "manual_annotations.csv")
    relevant_tags_file = os.path.join(dataset_path, "relevant_tags/relevant_tags.csv")
    
    if not os.path.exists(manual_annotations_file) or not os.path.exists(relevant_tags_file):
        print("Missing required files.")
        return
    
    manual_df = pd.read_csv(manual_annotations_file)
    relevant_df = pd.read_csv(relevant_tags_file)
    
    if "tags" not in manual_df.columns or "annotations" not in manual_df.columns or "tags" not in relevant_df.columns:
        print("Invalid file format.")
        return
    
    ground_truth = set(manual_df[manual_df["annotations"] == "a"]["tags"])
    predicted = set(relevant_df["tags"])
    
    true_positives = len(ground_truth & predicted)
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted
    
    precision = true_positives / (true_positives + len(false_positives)) if (true_positives + len(false_positives)) > 0 else 0
    recall = true_positives / (true_positives + len(false_negatives)) if (true_positives + len(false_negatives)) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"False Positives: {false_positives}")
    
    print(f"False Negatives: {false_negatives}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of the dataset to process")
    args = parser.parse_args()
    
    dataset_path = os.path.join("./data", args.dataset)
    if os.path.isdir(dataset_path):
        compute_metrics(dataset_path)
    else:
        print(f"Dataset {args.dataset} not found.")

if __name__ == "__main__":
    main()
