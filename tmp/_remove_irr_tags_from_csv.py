import pandas as pd


def load_and_modify_csv(input_path, output_path, column_to_remove):
    """
    Load a CSV, remove a column, and save the modified DataFrame.

    Parameters:
    - input_path: str, path to the input CSV file.
    - output_path: str, path to save the modified CSV file.
    - column_to_remove: str, name of the column to remove.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # Check if the column exists in the DataFrame
    if column_to_remove in df.columns:
        # Remove the specified column
        df = df.drop(columns=[column_to_remove])
        print(f"Column '{column_to_remove}' removed.")
    else:
        print(f"Column '{column_to_remove}' not found in the DataFrame.")

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Modified DataFrame saved to '{output_path}'.")


input_csv = "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/data/stanford-dogs-dataset/train_tags.csv"
output_csv = "/mnt/cephfs/home/gsarridis/projects/vb-mitigator/data/stanford-dogs-dataset/train_tags.csv"
column_name = "irrelevant_tags"

load_and_modify_csv(input_csv, output_csv, column_name)
