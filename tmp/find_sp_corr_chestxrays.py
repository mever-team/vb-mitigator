import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np  # Although not directly used for chi2, it's a common dependency


def analyze_spurious_correlations_from_csv(file_path):
    """
    Analyzes potential spurious correlations between medical findings and
    patient gender or age, loading data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the data.

    Returns:
        dict: A dictionary containing potential spurious correlations.
    """

    # 1. Parse the data from CSV
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

    # Rename columns to remove spaces if they exist (important for consistent access)
    df.columns = df.columns.str.strip()

    # Ensure 'Patient Age' is numeric
    # Use errors='coerce' to turn non-numeric values into NaN, then drop them
    df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
    df.dropna(
        subset=["Patient Age"], inplace=True
    )  # Remove rows where age couldn't be converted

    # 2. & 3. Extract relevant features and one-hot encode findings
    # Handle cases where 'Finding Labels' might be NaN or not a string
    df["Finding Labels"] = df["Finding Labels"].fillna("No Finding").astype(str)
    df["Finding Labels"] = df["Finding Labels"].apply(lambda x: x.split("|"))

    # Get all unique findings
    # Use a flat list comprehension for robustness
    all_findings = sorted(
        list(
            set([item.strip() for sublist in df["Finding Labels"] for item in sublist])
        )
    )

    # Filter out any empty strings that might result from splitting or bad data
    all_findings = [f for f in all_findings if f]

    # Create one-hot encoded columns for each finding
    for finding in all_findings:
        df[f"Finding_{finding}"] = df["Finding Labels"].apply(
            lambda x: 1 if finding in x else 0
        )

    # Prepare results dictionary
    spurious_correlations = {"gender_correlations": {}, "age_correlations": {}}

    # Analyze correlations with Gender
    print("\n--- Analyzing Correlations with Gender ---")
    for finding in all_findings:
        if f"Finding_{finding}" not in df.columns:
            print(f"Warning: Finding column 'Finding_{finding}' not found. Skipping.")
            continue

        # Filter out rows where Patient Gender is missing if any
        temp_df = df.dropna(subset=["Patient Gender"])
        if temp_df.empty:
            print(f"\nFinding: {finding} - No valid gender data to analyze.")
            continue

        contingency_table = pd.crosstab(
            temp_df["Patient Gender"], temp_df[f"Finding_{finding}"]
        )

        # Ensure valid table for chi-squared (at least 2x2 and total sum > 0)
        if (
            contingency_table.shape[0] > 1
            and contingency_table.shape[1] > 1
            and contingency_table.sum().sum() > 0
        ):
            try:
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                print(f"\nFinding: {finding}")
                print(contingency_table)
                print(f"Chi-squared statistic: {chi2:.2f}, P-value: {p:.3f}")
                if p < 0.05:  # Significance level
                    spurious_correlations["gender_correlations"][finding] = {
                        "p_value": p,
                        "contingency_table": contingency_table.to_dict("index"),
                    }
                    print(f"  --> Potentially correlated with Gender (p < 0.05)")
                else:
                    print(f"  --> No significant correlation with Gender (p >= 0.05)")
            except ValueError as e:
                print(
                    f"\nFinding: {finding} - Error during chi-squared test with Gender: {e}"
                )
                print(f"Contingency table:\n{contingency_table}")
        else:
            print(
                f"\nFinding: {finding} - Not enough valid data for chi-squared test with Gender."
            )

    # Analyze correlations with Age
    print("\n--- Analyzing Correlations with Age ---")
    # For age, discretize it into bins for chi-squared test.
    # You might want to adjust the bins based on your data distribution.
    # Ensure 'Patient Age' column is not empty after dropping NaNs
    if not df["Patient Age"].empty:
        # Create age bins, handling potential edge cases like all same age
        try:
            min_age = df["Patient Age"].min()
            max_age = df["Patient Age"].max()
            if min_age == max_age:
                print(
                    "Warning: All patients have the same age. Cannot create age bins for correlation."
                )
                df["Age_Bin"] = 0  # Assign a single bin if all ages are the same
            else:
                df["Age_Bin"] = pd.cut(
                    df["Patient Age"],
                    bins=5,
                    labels=False,
                    include_lowest=True,
                    duplicates="drop",
                )
                if (
                    df["Age_Bin"].nunique() < 2
                ):  # If binning results in too few unique bins (e.g., due to sparse data)
                    print(
                        "Warning: Age binning resulted in fewer than 2 unique bins. Using default value for now."
                    )
                    df["Age_Bin"] = pd.qcut(
                        df["Patient Age"],
                        q=min(df["Patient Age"].nunique(), 5),
                        labels=False,
                        duplicates="drop",
                    )
                    if df["Age_Bin"].nunique() < 2:
                        print(
                            "Warning: Still unable to create enough age bins. Skipping age correlation for this dataset."
                        )
                        return spurious_correlations  # Exit early if age binning fails
        except Exception as e:
            print(f"Error creating age bins: {e}. Skipping age correlation.")
            return spurious_correlations
    else:
        print("No valid patient age data available for correlation analysis.")
        return spurious_correlations

    for finding in all_findings:
        if f"Finding_{finding}" not in df.columns:
            print(f"Warning: Finding column 'Finding_{finding}' not found. Skipping.")
            continue

        # Filter out rows where Age_Bin is missing if any
        temp_df = df.dropna(subset=["Age_Bin"])
        if temp_df.empty:
            print(f"\nFinding: {finding} - No valid age bin data to analyze.")
            continue

        contingency_table = pd.crosstab(
            temp_df["Age_Bin"], temp_df[f"Finding_{finding}"]
        )

        # Ensure valid table for chi-squared (at least 2x2 and total sum > 0)
        if (
            contingency_table.shape[0] > 1
            and contingency_table.shape[1] > 1
            and contingency_table.sum().sum() > 0
        ):
            try:
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                print(f"\nFinding: {finding}")
                print(contingency_table)
                print(f"Chi-squared statistic: {chi2:.2f}, P-value: {p:.3f}")
                if p < 0.05:  # Significance level
                    spurious_correlations["age_correlations"][finding] = {
                        "p_value": p,
                        "contingency_table": contingency_table.to_dict("index"),
                    }
                    print(f"  --> Potentially correlated with Age (p < 0.05)")
                else:
                    print(f"  --> No significant correlation with Age (p >= 0.05)")
            except ValueError as e:
                print(
                    f"\nFinding: {finding} - Error during chi-squared test with Age: {e}"
                )
                print(f"Contingency table:\n{contingency_table}")
        else:
            print(
                f"\nFinding: {finding} - Not enough valid data for chi-squared test with Age."
            )

    return spurious_correlations


# --- Main execution when running the script ---
if __name__ == "__main__":
    csv_file_path = "/mnt/cephfs/home/gsarridis/datasets/chestxrays/Data_Entry_2017.csv"  # Make sure this CSV file is in the same directory as your script

    results = analyze_spurious_correlations_from_csv(csv_file_path)

    if results:
        print("\n--- Summary of Potentially Spurious Correlations (p < 0.05) ---")
        if results["gender_correlations"]:
            print("\nCorrelations with Gender:")
            for finding, details in results["gender_correlations"].items():
                print(f"  - {finding}: P-value = {details['p_value']:.3f}")
        else:
            print("\nNo significant correlations found with Gender.")

        if results["age_correlations"]:
            print("\nCorrelations with Age:")
            for finding, details in results["age_correlations"].items():
                print(f"  - {finding}: P-value = {details['p_value']:.3f}")
        else:
            print("\nNo significant correlations found with Age.")
    else:
        print("Analysis could not be completed due to data loading issues.")
