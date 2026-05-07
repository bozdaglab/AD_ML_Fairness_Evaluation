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


def demographic_parity(y_pred):
    dp_scores = []
    num_classes = 3
    for label in range(num_classes):
        binary_pred = (y_pred == label).astype(int)
        positive_rate = np.mean(binary_pred)
        dp_scores.append(positive_rate)
    return dp_scores


def equalized_odds(y_true, y_pred):
    num_classes = 3
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
    groups = list(data.keys())
    if len(groups) == 0:
        return []

    abs_per_positions = []

    num_lists = min(len(data[g]) for g in groups)

    for i in range(num_lists):
        lists = [data[group][i] for group in groups]
        abs_per_position = []

        for j in range(len(lists[0])):
            values = [lists[k][j] for k in range(len(groups))]
            max_pairwise_abs = max(abs(a - b) for a, b in combinations(values, 2))
            abs_per_position.append(max_pairwise_abs)

        abs_per_positions.append(max(abs_per_position))

    return abs_per_positions


def add_to_results(metric, output_data, results):
    for key, item in output_data.items():
        c = calculate_result(item)
        results.loc[len(results)] = [metric, key, c]


def get_top_features(file_path, preprocessed_columns):
    df = pd.read_csv(file_path)

    if "lasso" in file_path.lower() or "shap" in file_path.lower():
        df = df[df["features"].isin(preprocessed_columns)].copy()
        df["abs_scores"] = df["scores"].abs()
        top_feats = df.nlargest(100, "abs_scores")["features"].tolist()
    else:
        sorted_df = df.sort_values(by="scores", ascending=False)
        sorted_df_ = sorted_df[sorted_df["scores"] > 0].reset_index(drop=True)
        top_feats = sorted_df_[sorted_df_["features"].isin(preprocessed_columns)]["features"].tolist()
        top_feats = top_feats[:100]

    return top_feats


CLASS_NAME = "AD_Label"
path = str(Path(__file__).parent) + "/"

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
    [0, 1, 2],
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

preprocessed_columns = preprocessed.columns

label = preprocessed["file_name"].copy()

race_order = [
    "African_American",
    "Mexican_American",
    "Non-Hispanic_White"
]

feature_path = path + "Racial_Feature_Selection_Results/"

# Please uncomment the feature selection methods and models to run all combinations

racial_fs_methods = [
    # "SelectFromModel",
    "lasso",
    # "mutual_info",
    # "RFE",
    # "SelectBySingleFeaturePerformance",
    # "shap"
]

models = [
    ('RFC', RandomForestClassifier()),
    # ('SVM', SVC(kernel="rbf")),
    # ('Bayes', GaussianNB()),    
    # ('LR', LogisticRegression()),
    # ('KNN', KNeighborsClassifier()),
    # ('MLP', MLPClassifier(hidden_layer_sizes=(64, 32), solver="adam", max_iter=1000, early_stopping=True))
]

COVARIATES = ["Age", "ID_Language_Primary", "ID_Education", "ID_Gender"]
BASE_SEED = 42
N_RUNS = 50

for fs_method in racial_fs_methods:
    print(f"\nStarting feature selection method: {fs_method}")

    mex_file = feature_path + f"Mexican_American_{fs_method}_features_new_racial.csv"
    whi_file = feature_path + f"Non-Hispanic_White_{fs_method}_features_new_racial.csv"
    afr_file = feature_path + f"African_American_{fs_method}_features_new_racial.csv"

    try:
        mex = get_top_features(mex_file, preprocessed_columns)
        whi = get_top_features(whi_file, preprocessed_columns)
        afr = get_top_features(afr_file, preprocessed_columns)
    except FileNotFoundError:
        print(f"Skipping {fs_method} because one or more racial feature files are missing.")
        continue

    for cov in COVARIATES:
        if cov in preprocessed.columns and cov not in mex:
            mex.append(cov)
        if cov in preprocessed.columns and cov not in whi:
            whi.append(cov)
        if cov in preprocessed.columns and cov not in afr:
            afr.append(cov)

    feature_map = {
        "African_American": afr,
        "Mexican_American": mex,
        "Non-Hispanic_White": whi
    }

    base_preprocessed = preprocessed.copy()
    base_preprocessed_nofile = base_preprocessed.drop("file_name", axis=1)

    for model_name, model in models:
        print(f"  Running model: {model_name}")

        output_acc = defaultdict(list)
        output_f1mi = defaultdict(list)
        output_f1ma = defaultdict(list)
        output_bal_acc = defaultdict(list)
        output_dem_parity = defaultdict(list)
        output_eq_odds_tpr = defaultdict(list)
        output_eq_odds_fpr = defaultdict(list)

        for ind, main in enumerate(race_order):
            seed = BASE_SEED + ind
            random.seed(seed)
            np.random.seed(seed)

            group_idx = label[label == main].index
            new_train_data = base_preprocessed_nofile.loc[group_idx].copy()

            other_races = [r for r in race_order if r != main]
            new_other_test_data = [
                (grp, base_preprocessed_nofile.loc[label[label == grp].index].copy())
                for grp in other_races
            ]

            i = feature_map[main].copy()
            if "file_name" in i:
                i.remove("file_name")
            if CLASS_NAME in i:
                i.remove(CLASS_NAME)

            for run_idx in range(N_RUNS):
                new_label = new_train_data[CLASS_NAME]

                X_train, X_test, y_train, y_test = train_test_split(
                    new_train_data,
                    new_label,
                    test_size=0.20,
                    stratify=new_label,
                    random_state=seed + run_idx
                )

                scaler = MinMaxScaler().fit(X_train[i])

                X_train_scaled = pd.DataFrame(
                    scaler.transform(X_train[i]),
                    columns=i,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test[i]),
                    columns=i,
                    index=X_test.index
                )

                fitted_model = model.fit(X_train_scaled, y_train)
                y_pred = fitted_model.predict(X_test_scaled)

                output_f1mi[f"{main}_{model_name}"].append(f1_score(y_test, y_pred, average="micro"))
                output_f1ma[f"{main}_{model_name}"].append(f1_score(y_test, y_pred, average="macro"))
                output_acc[f"{main}_{model_name}"].append(accuracy_score(y_test, y_pred))
                output_bal_acc[f"{main}_{model_name}"].append(balanced_accuracy_score(y_test, y_pred))

                dem_parity = demographic_parity(y_pred)
                output_dem_parity[f"{main}_{model_name}"].append(dem_parity)

                tpr, fpr = equalized_odds(y_test, y_pred)
                output_eq_odds_tpr[f"{main}_{model_name}"].append(tpr)
                output_eq_odds_fpr[f"{main}_{model_name}"].append(fpr)

                for name_k, test_data in new_other_test_data:
                    y_true = test_data[CLASS_NAME]
                    X_other = test_data[i].copy()

                    X_other_scaled = pd.DataFrame(
                        scaler.transform(X_other),
                        columns=i,
                        index=X_other.index
                    )

                    pred = fitted_model.predict(X_other_scaled)

                    output_f1mi[f"{main}_{name_k}_{model_name}"].append(f1_score(y_true, pred, average="micro"))
                    output_f1ma[f"{main}_{name_k}_{model_name}"].append(f1_score(y_true, pred, average="macro"))
                    output_acc[f"{main}_{name_k}_{model_name}"].append(accuracy_score(y_true, pred))
                    output_bal_acc[f"{main}_{name_k}_{model_name}"].append(balanced_accuracy_score(y_true, pred))

                    dem_parity = demographic_parity(pred)
                    output_dem_parity[f"{main}_{name_k}_{model_name}"].append(dem_parity)

                    tpr, fpr = equalized_odds(y_true, pred)
                    output_eq_odds_tpr[f"{main}_{name_k}_{model_name}"].append(tpr)
                    output_eq_odds_fpr[f"{main}_{name_k}_{model_name}"].append(fpr)

        dem_items = list(output_dem_parity.items())
        tpr_items = list(output_eq_odds_tpr.items())
        fpr_items = list(output_eq_odds_fpr.items())


        afr_keys = [k for k in output_dem_parity if k.startswith("African_American_")]
        mex_keys = [k for k in output_dem_parity if k.startswith("Mexican_American_")]
        whi_keys = [k for k in output_dem_parity if k.startswith("Non-Hispanic_White_")]

        output_dem_parity1 = compute_abs_diff({k: output_dem_parity[k] for k in afr_keys}) if afr_keys else []
        output_eq_odds_tpr1 = compute_abs_diff({k: output_eq_odds_tpr[k] for k in afr_keys}) if afr_keys else []
        output_eq_odds_fpr1 = compute_abs_diff({k: output_eq_odds_fpr[k] for k in afr_keys}) if afr_keys else []

        output_dem_parity2 = compute_abs_diff({k: output_dem_parity[k] for k in mex_keys}) if mex_keys else []
        output_eq_odds_tpr2 = compute_abs_diff({k: output_eq_odds_tpr[k] for k in mex_keys}) if mex_keys else []
        output_eq_odds_fpr2 = compute_abs_diff({k: output_eq_odds_fpr[k] for k in mex_keys}) if mex_keys else []

        output_dem_parity3 = compute_abs_diff({k: output_dem_parity[k] for k in whi_keys}) if whi_keys else []
        output_eq_odds_tpr3 = compute_abs_diff({k: output_eq_odds_tpr[k] for k in whi_keys}) if whi_keys else []
        output_eq_odds_fpr3 = compute_abs_diff({k: output_eq_odds_fpr[k] for k in whi_keys}) if whi_keys else []

        df_bal_acc = pd.DataFrame(output_bal_acc) 
        df_f1ma = pd.DataFrame(output_f1ma) 
        df_bal_acc.to_csv(f'Results_Racial_{fs_method}_{model_name}_balanced_accuracy.csv', index=False) 
        df_f1ma.to_csv(f'Results_Racial_{fs_method}_{model_name}_f1_macro.csv', index=False)

        results = pd.DataFrame(columns=["Metric", "Race", "Value"])

        add_to_results("Balanced Accuracy", output_bal_acc, results)
        add_to_results("F1-Macro Accuracy", output_f1ma, results)

        if len(output_dem_parity1) > 0:
            results.loc[len(results)] = ["Demographic Parity (DP)", "Overall_African", calculate_result(output_dem_parity1)]
            results.loc[len(results)] = ["Equal Opportunity (EOp)", "Overall_African", calculate_result(output_eq_odds_tpr1)]
            results.loc[len(results)] = [
                "Equalized Odds Difference (EOd)",
                "Overall_African",
                calculate_result(list(abs(np.array(output_eq_odds_tpr1) + np.array(output_eq_odds_fpr1))))
            ]

        if len(output_dem_parity2) > 0:
            results.loc[len(results)] = ["Demographic Parity (DP)", "Overall_Mexican", calculate_result(output_dem_parity2)]
            results.loc[len(results)] = ["Equal Opportunity (EOp)", "Overall_Mexican", calculate_result(output_eq_odds_tpr2)]
            results.loc[len(results)] = [
                "Equalized Odds Difference (EOd)",
                "Overall_Mexican",
                calculate_result(list(abs(np.array(output_eq_odds_tpr2) + np.array(output_eq_odds_fpr2))))
            ]

        if len(output_dem_parity3) > 0:
            results.loc[len(results)] = ["Demographic Parity (DP)", "Overall_White", calculate_result(output_dem_parity3)]
            results.loc[len(results)] = ["Equal Opportunity (EOp)", "Overall_White", calculate_result(output_eq_odds_tpr3)]
            results.loc[len(results)] = [
                "Equalized Odds Difference (EOd)",
                "Overall_White",
                calculate_result(list(abs(np.array(output_eq_odds_tpr3) + np.array(output_eq_odds_fpr3))))
            ]

        results.to_csv(f"Results_Racial_{fs_method}_{model_name}.csv", index=False)
        print(f"Saved: Results_Racial_{fs_method}_{model_name}.csv")