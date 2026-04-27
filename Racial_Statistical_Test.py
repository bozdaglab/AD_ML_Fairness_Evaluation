import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import itertools

# Helper to compute Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*np.std(x, ddof=1) ** 2 + (ny - 1)*np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

# Prepare a function to extract comparison details
def extract_groups(colname):
    parts = colname.split("_")
    train_group = parts[0]
    test_group = "_".join(parts[2:]) if "RFC" not in colname else parts[0]
    if "RFC" in colname:
        test_group = train_group
    return train_group, test_group

# Function to compute Welch's t-test and Cohen's d for all metrics
def analyze_metric(df, metric_name):
    results = []
    for col in df.columns:
        train_group, test_group = extract_groups(col)
        values = df[col].dropna()
        # Split values into train vs. test group metrics
        if "RFC" in col:
            ref_values = values
            ref_key = (train_group, train_group)
        else:
            comp_values = values
            comp_key = (train_group, test_group)
            if ref_key[0] == comp_key[0]:
                t_stat, p_value = ttest_ind(ref_values, comp_values, equal_var=False)
                d_value = cohens_d(ref_values, comp_values)
                results.append({
                    "Train Group": train_group,
                    "Test Group": test_group,
                    "Metric": metric_name,
                    "p-value": p_value,
                    "Cohen's d": d_value
                })
    return results

def strip_last_word_if_more_than_4(col):
    parts = col.split("_")
    if len(parts) > 4:
        return "_".join(parts[:-1])
    return col


# Load data-do change the column rames _RFC should be on only one column
balanced_df = pd.read_csv("Results_Racial_lasso_RFC_balanced_accuracy.csv")
macro_f1_df = pd.read_csv("Results_Racial_lasso_RFC_f1_macro.csv")
balanced_df.columns = [strip_last_word_if_more_than_4(c) for c in balanced_df.columns]
macro_f1_df.columns = [strip_last_word_if_more_than_4(c) for c in macro_f1_df.columns]

# Run analysis for all three metrics
# results_accuracy = analyze_metric(accuracy_df, "Accuracy")
results_balanced = analyze_metric(balanced_df, "Balanced Accuracy")
results_macro_f1 = analyze_metric(macro_f1_df, "Macro F1")

# Combine results into single DataFrame
all_results =  results_balanced + results_macro_f1
# all_results = results_accuracy + results_balanced + results_macro_f1
results_df = pd.DataFrame(all_results)

# Pivot for final table format
final_table = results_df.pivot(index=["Train Group", "Test Group"], columns="Metric", values=["p-value", "Cohen's d"])
final_table.columns = [f"{col[1]}_{col[0]}" for col in final_table.columns]
final_table.reset_index(inplace=True)

# Save to CSV
final_table.to_csv("Results_Racial_lasso_RFC_welch_test_results.csv", index=False)
