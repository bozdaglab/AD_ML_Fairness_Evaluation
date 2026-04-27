import pandas as pd
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from itertools import combinations
from statistics import mean
import random
from pathlib import Path


def calculate_result(inp) -> str:
    try:
        return f"{round(statistics.mean(inp), 3)}+-{round(statistics.stdev(inp), 3)}"
    except Exception:
        return f"{round(statistics.mean(inp), 3)}+-{round(np.std(inp), 3)}"


def demographic_parity(y_pred, num_classes=3):
    dp_scores = []
    for label in range(num_classes):
        binary_pred = (y_pred == label).astype(int)
        positive_rate = np.mean(binary_pred)
        dp_scores.append(positive_rate)
    return dp_scores


def equalized_odds(y_true, y_pred, num_classes=3):
    tpr_scores = []
    fpr_scores = []

    for label in range(num_classes):
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_scores.append(tpr)

        fp = np.sum((y_pred == label) & (y_true != label))
        tn = np.sum((y_pred != label) & (y_true != label))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_scores.append(fpr)

    return tpr_scores, fpr_scores


def compute_abs_diff(data):
    result = []
    mad_per_positions = []
    abs_per_positions = []

    groups = list(data.keys())
    if len(groups) == 0:
        return [], [], []

    num_lists = len(data[groups[0]])

    for i in range(num_lists):
        lists = [data[group][i] for group in groups]

        mad_per_position = []
        abs_per_position = []

        for j in range(len(lists[0])):
            values = [lists[k][j] for k in range(len(groups))]
            avg = mean(values)

            mad = max(abs(x - avg) for x in values) / len(values)
            max_pairwise_abs = max(abs(a - b) for a, b in combinations(values, 2))

            mad_per_position.append(mad)
            abs_per_position.append(max_pairwise_abs)

        mad_per_positions.append(mad_per_position)
        abs_per_positions.append(max(abs_per_position))
        result.append(max(abs_per_positions))

    return result, mad_per_positions, abs_per_positions


def add_to_results(metric, output_data, results_df):
    for key, item in output_data.items():
        c = calculate_result(item)
        results_df.loc[len(results_df)] = [metric, key, c]


def preprocess(path):
    preprocessed = pd.read_csv(path + "final_preprocessed_HABS_new.csv")
    preprocessed.drop("Unnamed: 0", axis=1, inplace=True, errors="ignore")

    preprocessed["Amyloid_Pos"] = (
        preprocessed["01_AB_FBB_Global_SUVR"] > 1.08
    ).astype(int)

    preprocessed["AD_Label"] = np.select(
        [
            (preprocessed["CDX_Cog"] == 0) & (preprocessed["Amyloid_Pos"] == 0),
            (preprocessed["CDX_Cog"] == 1) & (preprocessed["Amyloid_Pos"] == 1),
            (preprocessed["CDX_Cog"] == 2) & (preprocessed["Amyloid_Pos"] == 1),
        ],
        [
            0,  # Normal
            1,  # MCI due to AD
            2   # AD dementia
        ],
        default=np.nan
    )

    preprocessed = preprocessed.loc[preprocessed["AD_Label"].notna()].copy()
    preprocessed["AD_Label"] = preprocessed["AD_Label"].astype(int)
    preprocessed.reset_index(drop=True, inplace=True)

    preprocessed.drop(columns=["CDX_Cog", "Amyloid_Pos", "01_AB_FBB_Global_SUVR", "01_AB_FBB_pos"], inplace=True, errors="ignore")

    preprocessed.drop(
        columns=preprocessed.filter(regex=r"^(ID_Hispanic|ID_Race)").columns,
        inplace=True,
        errors="ignore"
    )


    return preprocessed


CLASS_NAME = "AD_Label"
path = str(Path(__file__).parent) + "/"

fs_file_names = [
    # "combined_lasso_features_new.csv",
    # "combined_mutual_info_features_new.csv",
    # "combined_RFE_features_new.csv",
    # "combined_SelectFromModel_features_new.csv",
    "combined_SelectBySingleFeaturePerformance_features_new.csv",
    # "combined_shap_features_new.csv"
]

models = [
    ('RFC', RandomForestClassifier()),
#     ('SVM', SVC(kernel="rbf")),
#     ('Bayes', GaussianNB()),    
#     ('LR', LogisticRegression()),
#     ('KNN', KNeighborsClassifier()),
#     ('MLP', MLPClassifier(hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000, early_stopping=True))
]

BASE_SEED = 42
N_RUNS = 50
TOP_K = 100
COVARIATES = ["Age", "ID_Language_Primary", "ID_Education", "ID_Gender"]

for fs_file_name in fs_file_names:
    print(f"\nProcessing FS file: {fs_file_name}")

    preprocessed = preprocess(path)
    # print(preprocessed.groupby(["file_name", "AD_Label"]).size())
    fs_df = pd.read_csv(path + 'Combined_Feature_Selection_Results/' + fs_file_name)

    sorted_df = fs_df.sort_values(by='scores', ascending=False)
    preprocessed_columns = preprocessed.columns

    feature_values = sorted_df[sorted_df['features'].isin(preprocessed_columns)][['features', 'scores']].copy()

    if "lasso" in fs_file_name.lower() or "shap" in fs_file_name.lower():
        feature_values['abs_scores'] = feature_values['scores'].abs()
        union_feature = feature_values.nlargest(TOP_K, 'abs_scores')['features'].tolist()

    else:
        union_feature = feature_values.nlargest(TOP_K, 'scores')['features'].tolist()

    for cov in COVARIATES:
        if cov in preprocessed.columns and cov not in union_feature:
            union_feature.append(cov)

    # Remove target / group cols if accidentally included
    union_feature = [f for f in union_feature if f not in [CLASS_NAME, "file_name"]]

    for model_name, model in models:
        print(f"  Running model: {model_name}")

        output_acc = defaultdict(list)
        output_f1mi = defaultdict(list)
        output_f1ma = defaultdict(list)
        output_bal_acc = defaultdict(list)
        output_dem_parity = defaultdict(list)
        output_eq_odds_tpr = defaultdict(list)
        output_eq_odds_fpr = defaultdict(list)

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            random.seed(seed)
            np.random.seed(seed)

            new_train_data = preprocessed.copy()
            new_label = new_train_data[CLASS_NAME]

            stratify_label = (
                new_train_data[CLASS_NAME].astype(str) + "_" +
                new_train_data["file_name"].astype(str)
            )

            X_train, X_test, y_train, y_test = train_test_split(
                new_train_data,
                new_label,
                test_size=0.30,
                stratify=stratify_label,
                random_state=seed
            )

            X_train = X_train.copy()
            X_test = X_test.copy()

            X_train.drop("file_name", axis=1, inplace=True, errors="ignore")

            unique_values = X_test["file_name"].unique()
            keeper = defaultdict(dict)

            for value in unique_values:
                val = X_test[X_test["file_name"] == value].copy()
                keeper[value]["x_test"] = val
                keeper[value]["y_test"] = y_test[y_test.index.isin(val.index)]

            scaler = MinMaxScaler().fit(X_train[union_feature])
            X_train_scaled = pd.DataFrame(
                scaler.transform(X_train[union_feature]),
                columns=union_feature,
                index=X_train.index
            )

            model.fit(X_train_scaled, y_train)

            for value in unique_values:
                X_group = keeper[value]["x_test"][union_feature]
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_group),
                    columns=union_feature,
                    index=X_group.index
                )

                y_true_group = keeper[value]["y_test"]
                y_pred = model.predict(X_test_scaled)

                output_f1mi[value].append(f1_score(y_true_group, y_pred, average='micro'))
                output_f1ma[value].append(f1_score(y_true_group, y_pred, average='macro'))
                output_acc[value].append(accuracy_score(y_true_group, y_pred))
                output_bal_acc[value].append(balanced_accuracy_score(y_true_group, y_pred))

                dem_parity = demographic_parity(y_pred)
                output_dem_parity[value].append(dem_parity)

                tpr, fpr = equalized_odds(y_true_group, y_pred)
                output_eq_odds_tpr[value].append(tpr)
                output_eq_odds_fpr[value].append(fpr)

        # Fairness across groups
        _, _, output_dem_parity_abs = compute_abs_diff(output_dem_parity)
        _, _, output_eq_odds_tpr_abs = compute_abs_diff(output_eq_odds_tpr)
        _, _, output_eq_odds_fpr_abs = compute_abs_diff(output_eq_odds_fpr)


        results = pd.DataFrame(columns=["Metric", "Race", "Value"])

        add_to_results("Accuracy", output_acc, results)
        add_to_results("Balanced Accuracy", output_bal_acc, results)
        add_to_results("F1-Micro Accuracy", output_f1mi, results)
        add_to_results("F1-Macro Accuracy", output_f1ma, results)

        results.loc[len(results)] = [
            "Demographic Parity (DP)", "ABS", calculate_result(output_dem_parity_abs)
        ]
        results.loc[len(results)] = [
            "Equal Opportunity (EOp)", "ABS", calculate_result(output_eq_odds_tpr_abs)
        ]
        results.loc[len(results)] = [
            "Equalized Odds Difference (EOd)", "ABS",
            calculate_result(list(np.array(output_eq_odds_tpr_abs) + np.array(output_eq_odds_fpr_abs)))
        ]

        fs_short = fs_file_name.replace(".csv", "").replace("combined_", "").replace("_new", "")
        out_file = f"Results_Baseline_{fs_short}_{model_name}.csv"
        results.to_csv(out_file, index=False)

        print(f"  Saved: {out_file}")