import logging
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from boruta import BorutaPy
from feature_engine.selection import SelectByShuffling, SelectBySingleFeaturePerformance
from genetic_selection import GeneticSelectionCV
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import ExtraTreeRegressor

from feature_selection_type import FeatureSelectionType
from set_logger import set_log_config
from settings import COLUMNS_FOR_GROUPING, CSV_PATH, GROUP_ID

set_log_config()
logger = logging.getLogger()


def save_dict_to_pd(diagnos_label: Dict, name: str) -> None:
    f = defaultdict(list)
    for i, k in diagnos_label.items():
        f["_".join(k)].append(i)
    pd.DataFrame({"patient": f.keys(), "values": f.values()}).to_csv(f"{name}.csv")


def search_dictionary(methods_features: Dict, thr: int = 2) -> List[str]:
    count_features = Counter()
    for features in methods_features.values():
        count_features.update(features)
    return [feat for feat, val in count_features.items() if val > thr]


def ratio(number: int) -> int:
    return round(number * 60 / 100) - 1


def lower_upper_bound(new_alpha: float, mulitply: int = 2, range_value=6) -> np.array:
    large = new_alpha * mulitply
    lower_bound = abs(new_alpha - large)
    upper_bound = new_alpha + large
    return np.linspace(lower_bound, upper_bound, range_value)


def feature_combination(data: pd.DataFrame) -> List[Tuple[str, str]]:
    return combinations(data.columns, 2)


def interview_time(data: pd.DataFrame) -> pd.DataFrame:
    interview_duration = []
    for start, end in zip(data.Interview_Start, data.Interview_End):
        if (
            data.Interview_Start.dtype.name == "object"
            or data.Interview_End.dtype.name == "object"
        ):
            logger.info("convert {start} and {end} interview time.")
            end_time = datetime.strptime(end, "%I:%M:%S %p")
            start_time = datetime.strptime(start, "%I:%M:%S %p")
            delta = (end_time - start_time).seconds
            hour, minutes = divmod(delta, 3600)
            delta = f"{hour}:{int(minutes / 60)}"
        else:
            logger.info(
                "there is no need to convert the time, as it has already converted"
            )
            delta = end - start
        interview_duration.append(delta)
    data.drop(["Interview_Start", "Interview_End"], axis=1, inplace=True)
    data["interview_duration"] = interview_duration
    return data


def write_xlsx_dict(file: str, inp_list: Dict) -> None:
    with pd.ExcelWriter(f"{CSV_PATH}/{file}.xlsx") as writer:
        for name, fun in inp_list.items():
            if name not in ["data"]:
                try:
                    fun.to_excel(writer, sheet_name=name)
                except:
                    pd.DataFrame(fun).to_excel(writer, sheet_name=name)


def write_feature_stats_excel(
    data: pd.DataFrame, columns: List[str], method: str
) -> None:
    with pd.ExcelWriter(f"{CSV_PATH}/features_stat_{method}.xlsx") as writer:
        for group in data.groupby(COLUMNS_FOR_GROUPING):
            name = "_".join(str(i) for i in group[0])
            data_features = group[1][columns].describe()
            data_features.to_excel(writer, sheet_name=name)


def combine_cohort(all_cohorts: dict, save_file: bool = True) -> pd.DataFrame:
    shared_column = search_dictionary(all_cohorts, thr=len(all_cohorts) - 1)
    df_combined = pd.DataFrame(columns=shared_column)
    for key, data in all_cohorts.items():
        # if key in ["HD 3 Mexican American 50+ Request 299", "HD 3 Non-Hispanic White 50+ Request 299", "HD 1 African American 50+ Request 299"]:
        df_combined = df_combined.append(data[shared_column])
    df_combined.reset_index(drop=True, inplace=True)
    if save_file:
        df_combined.to_csv(f"{CSV_PATH}/combined_dataset_april1.csv", index=False)
    return df_combined


def load_missing_method(imputer_name: str, n_neighbors: int = 3):
    if imputer_name == "KNNImputer":
        return KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
    elif imputer_name == "IterativeImputer":
        return IterativeImputer(
            max_iter=5, estimator=ExtraTreeRegressor(), keep_empty_features=True
        )
    return None


def load_models1(
    feature_selection_type: str,
    mlmodel: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_number: int,
) -> [RFE, SelectFromModel, SequentialFeatureSelector]:
    if feature_selection_type == FeatureSelectionType.RFE.name:
        return RFE(
            estimator=mlmodel,
            n_features_to_select=feature_number,
            step=10,
            verbose=5,
        )

    elif feature_selection_type == FeatureSelectionType.SelectFromModel.name:
        return SelectFromModel(
            estimator=mlmodel,
            max_features=feature_number,
            threshold="-0.0000000000000000000000000000000000001*median",
        )

    elif feature_selection_type == FeatureSelectionType.SequentialFeatureSelector.name:
        return SequentialFeatureSelector(
            estimator=mlmodel,
            k_features=feature_number,
            forward=True,
            verbose=2,
            cv=5,
            n_jobs=2,
            scoring="r2",
        )


def group_by_given_column(data: pd.DataFrame, columns_to_group: List[str]):
    return (
        data.groupby(GROUP_ID)[columns_to_group]
        .apply(lambda x: x.values.tolist() if len(x) > 1 else None)
        .dropna()
    )


def load_models2(
    feature_selection_type: str,
    mlmodel: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    max_iter,
) -> [
    SelectBySingleFeaturePerformance,
    SelectByShuffling,
    GeneticSelectionCV,
    BorutaPy,
]:
    if (
        feature_selection_type
        == FeatureSelectionType.SelectBySingleFeaturePerformance.name
    ):
        return SelectBySingleFeaturePerformance(
            estimator=mlmodel,
            scoring="f1_macro",
            threshold=0.6,
            cv=5,
        )

    elif feature_selection_type == FeatureSelectionType.SelectByShuffling.name:
        return SelectByShuffling(
            estimator=mlmodel,
            scoring="f1_macro",
            threshold=0.06,
            cv=5,
        )

    elif feature_selection_type == FeatureSelectionType.GeneticSelectionCV.name:
        return GeneticSelectionCV(
            estimator=mlmodel,
            cv=5,
            scoring="f1_macro",
            max_features=max_iter,
            n_population=max_iter,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=max_iter,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            n_gen_no_change=10,
            caching=True,
            n_jobs=-1,
        )

    elif feature_selection_type == FeatureSelectionType.BorutaPy.name:
        return BorutaPy(
            estimator=mlmodel,
            n_estimators=mlmodel.n_estimators,
            verbose=2,
            max_iter=max_iter,
        )


def test_individual(data_path, file, preprocess=True):
    from pre_process import label_encoding, pre_processing

    if not file.endswith(".xlsx"):
        data = pd.read_csv(f"{data_path}/{file}")
        data.rename(columns={"PTRACCAT":"file_name", "DX":"CDR_Diagnosis_1"}, inplace=True)
        x = data[data.columns[:61]]
        x = x.iloc[data["CDR_Diagnosis_1"].dropna().index]
        x.reset_index(drop=True, inplace=True)
        x = x.iloc[[idx for idx, i in enumerate(x["file_name"]) if i not in ["Unknown", "More than one", "Am Indian/Alaskan", "Hawaiian/Other PI"]]]
        
        prune_dataset = []
        for pat in x.groupby("PTID"):
            prune_dataset.append(pat[1].iloc[-1:].values[0])
        dataset = pd.DataFrame(prune_dataset, columns=x.columns)
        if preprocess:
            # data = interview_time(data)
            # data = label_encoding(all_cohorts=x)
            data = pre_processing(dataset)
    return data
    