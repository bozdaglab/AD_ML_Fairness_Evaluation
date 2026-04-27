import statistics
import pandas as pd
import numpy as np
import statistics
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  GridSearchCV
from collections import defaultdict
from itertools import combinations
from scipy.ndimage import gaussian_filter
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from statistics import mean
from pathlib import Path


def calculate_result(inp) -> str:
    try:
        return f"{round(statistics.mean(inp), 3)}+-{round(statistics.stdev(inp), 3)}"
    except AssertionError:
        return f"{round(statistics.mean(inp), 3)}+-{round(np.std(inp), 3)}"

def calculate_result_diff(inp) -> str:
    try:
        return round(statistics.mean(inp), 3)
    except AssertionError:
        return round(statistics.mean(inp), 3)

def demographic_parity(y_pred):
    dp_scores = []
    num_classes=3
    for label in range(num_classes):
    
        binary_pred = (y_pred == label).astype(int)
        positive_rate = np.mean(binary_pred)
        dp_scores.append(positive_rate)

    return dp_scores

def equalized_odds(y_true, y_pred):
    num_classes=3
    tpr_scores = []
    fpr_scores = []
    num_classes=3

    for label in range(num_classes):
        # TPR for the current label within the group
        tp = np.sum((y_pred == label) & (y_true == label))
        fn = np.sum((y_pred != label) & (y_true == label))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_scores.append(tpr)

        # FPR for the current label within the group
        fp = np.sum((y_pred == label) & (y_true != label))
        tn = np.sum((y_pred != label) & (y_true != label))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_scores.append(fpr)
    
    return tpr_scores, fpr_scores


from collections import Counter
import numpy as np

def custom_weighted_training(X, y, sensitive_attr):

    model = RandomForestClassifier(class_weight="balanced")

    g = sensitive_attr.to_numpy()
    y_arr = np.asarray(y)

    group_counts = Counter(g)
    class_counts = Counter(y_arr)

    total = len(y_arr)
    n_groups = len(np.unique(g))
    n_classes = len(np.unique(y_arr))

    group_w = np.array([
        total / (n_groups * group_counts[gi])
        for gi in g
    ])

    class_w = np.array([
        total / (n_classes * class_counts[yi])
        for yi in y_arr
    ])

    w = np.sqrt(group_w * class_w)

    w = w / np.mean(w)

    model.fit(X, y, sample_weight=w)
    return model



def compute_abs_diff(data):
    result = []
    mad_per_positions = []
    abs_per_positions =[] 
    groups = list(data.keys())
    num_lists = len(data[groups[0]])
    for i in range(num_lists):
        # Collect the lists for all groups
        lists = [data[group][i] for group in groups]
        mad_per_position = []
        abs_per_position = []
        # For each position in the lists
        for j in range(len(lists[0])):
            # Extract values at the same position for all groups
            values = [lists[k][j] for k in range(len(groups))]
            
            # Compute the mean
            avg = mean(values)
            
            # Compute MAD
            mad = max(abs(x - avg) for x in values) / len(values)
            sum_abs_values = max(abs(a - b) for a, b in combinations(values, 2))

            mad_per_position.append(mad)
            abs_per_position.append(sum_abs_values)

        mad_per_positions.append(mad_per_position)
        abs_per_positions.append(max(abs_per_position))
        result.append(max(abs_per_positions))
    return result, mad_per_positions, abs_per_positions


def add_to_results(metric, output_data):
    for key, item in output_data.items():
        c = calculate_result(item)
        results.loc[len(results)] = [metric, key, c]



CLASS_NAME="AD_Label"
fs_file_name = "Pre_Processing_Feature_Selection"

path = str(Path(__file__).parent) + "/"
preprocessed = pd.read_csv(path+ "final_preprocessed_HABS_new.csv")
preprocessed.drop("Unnamed: 0", axis=1,inplace=True)


preprocessed["Amyloid_Pos"] = (
    preprocessed["01_AB_FBB_Global_SUVR"] > 1.08
).astype(int)


preprocessed["AD_Label"] = np.select(
    [
        (preprocessed["CDX_Cog"] == 0) & (preprocessed["Amyloid_Pos"] == 0) ,
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

preprocessed.drop("CDX_Cog", axis=1,inplace=True)
preprocessed.drop("Amyloid_Pos", axis=1,inplace=True)
preprocessed.drop("01_AB_FBB_Global_SUVR", axis=1,inplace=True)
preprocessed.drop("01_AB_FBB_pos", axis=1,inplace=True)


a = pd.read_csv(path + fs_file_name +'.csv')

sorted_df = a.sort_values(by=['count','scores'], ascending=False)
preprocessed_columns= preprocessed.columns
feature_values = sorted_df[sorted_df['features'].isin(preprocessed_columns)][['features', 'scores']]
union_feature = feature_values["features"].to_list()
union_feature = union_feature[0:100]

COVARIATES = ["Age", "ID_Language_Primary", "ID_Education", "ID_Gender"]

for cov in COVARIATES:
    if cov in preprocessed.columns and cov not in union_feature:
        union_feature.append(cov)


output = defaultdict(list)
output_acc = defaultdict(list)
output_f1mi = defaultdict(list)
output_f1ma = defaultdict(list)
output_bal_acc = defaultdict(list)
output_dem_parity = defaultdict(list)
output_eq_odds = defaultdict(list)
output_eq_odds_tpr = defaultdict(list)
output_eq_odds_fpr = defaultdict(list)

df_acc = pd.DataFrame(output_acc)
df_bal_acc = pd.DataFrame(output_bal_acc)
df_f1mi = pd.DataFrame(output_f1mi)
df_f1ma = pd.DataFrame(output_f1ma)
df_dem_parity = pd.DataFrame(output_dem_parity)
df_eq_odds_tpr = pd.DataFrame(output_eq_odds_tpr)
df_eq_odds_fpr = pd.DataFrame(output_eq_odds_fpr)


output = defaultdict(list)
label = preprocessed["file_name"]
labels_list = label.unique()
new_train_data = preprocessed
best_model = None
best_accuracy = 0
BASE_SEED = 42
res = []


print("starting.........")
for run_idx in range(50):
    SEED = BASE_SEED + run_idx
    random.seed(SEED)
    np.random.seed(SEED)
    print(run_idx)
    new_label = new_train_data[CLASS_NAME]

    stratify_label= (
            new_train_data[CLASS_NAME].astype(str) + "_" + new_train_data["file_name"].astype(str)
        )
    X_train, X_test, y_train, y_test= train_test_split(
            new_train_data, new_label, test_size=0.30, stratify=stratify_label, random_state=SEED
        )
    
    X_train.drop("file_name", axis=1, inplace=True)


    unique_values = X_test["file_name"].unique()
    keeper = defaultdict(defaultdict)
    for value in unique_values:
        val = X_test[X_test["file_name"] == value]
        keeper[value]["x_test"] = val
        keeper[value]["y_test"] = y_test[y_test.index.isin(val.index)]

    i = union_feature
    if "file_name" in i:
        i.remove("file_name")
    if CLASS_NAME in i:
        i.remove(CLASS_NAME)

    scaler = MinMaxScaler().fit(X_train[i])
    X_train= pd.DataFrame(scaler.transform(X_train[i]),columns=i, index=X_train.index)


    X_train_sensitive_group = new_train_data["file_name"].loc[X_train.index]

    best_model = custom_weighted_training(X_train[i], y_train, X_train_sensitive_group)
    
    for value in unique_values:
            X_group=keeper[value]["x_test"][i]
            X_test1 = pd.DataFrame(scaler.transform(X_group),columns=i, index=X_group.index)

               
            y_pred= best_model.predict(X_test1)

            accuracy = f1_score(keeper[value]["y_test"], y_pred, average='micro')
            print(f'{value} Test Accuracy_f1mi: {accuracy}')
            output_f1mi[f'{value}'].append(accuracy)
            accuracy = f1_score(keeper[value]["y_test"], y_pred, average='macro')
            print(f'{value} Test Accuracy_f1ma: {accuracy}')
            output_f1ma[f'{value}'].append(accuracy)
            accuracy = accuracy_score(keeper[value]["y_test"], y_pred)
            print(f'{value} Test Accuracy: {accuracy}')
            output_acc[f'{value}'].append(accuracy)
            accuracy = balanced_accuracy_score(keeper[value]["y_test"], y_pred)
            print(f'{value} Test Balanced Accuracy: {accuracy}')
            output_bal_acc[f'{value}'].append(accuracy)
            # Fairness metrics: Demographic Parity and Equalized Odds
            dem_parity = demographic_parity(y_pred)
            print(f'{value} Demographic Parity Difference: {dem_parity}')
            output_dem_parity[f'{value}'].append(dem_parity)

            tpr, fpr = equalized_odds(keeper[value]["y_test"],y_pred)
            print(f'{value} Equalized Odds TPR: {tpr}')
            output_eq_odds_tpr[f'{value}'].append(tpr)
            print(f'{value} Equalized Odds FPR: {fpr}')
            output_eq_odds_fpr[f'{value}'].append(fpr)

output_dem_parity1, output_dem_parity, output_dem_parity_abs = compute_abs_diff(output_dem_parity)
output_eq_odds_tpr1, output_eq_odds_tpr, output_eq_odds_tpr_abs = compute_abs_diff(output_eq_odds_tpr)
output_eq_odds_fpr1, output_eq_odds_fpr, output_eq_odds_fpr_abs = compute_abs_diff(output_eq_odds_fpr)


results = pd.DataFrame(columns=["Metric", "Race", "Value"])
# Accuracy
add_to_results("Accuracy", output_acc)
# Balanced Accuracy
add_to_results("Balanced Accuracy", output_bal_acc)
# F1-Micro Accuracy
add_to_results("F1-Micro Accuracy", output_f1mi)
# F1-Macro Accuracy
add_to_results("F1-Macro Accuracy", output_f1ma)
# Demographic Parity Difference
results.loc[len(results)] = ["Demographic Parity (DP)", "ABS", calculate_result(output_dem_parity_abs)]

results.loc[len(results)] = ["Equal Opportunity (EOp)", "ABS", calculate_result(output_eq_odds_tpr_abs)]

results.loc[len(results)] = ["Equalized Odds Difference (EOd) ", "ABS", calculate_result(list(np.array(output_eq_odds_tpr_abs) + np.array(output_eq_odds_fpr_abs)))]

results.to_csv("Results_Pre+In-Processing.csv", index=False)

print("Results saved to results.xlsx")
