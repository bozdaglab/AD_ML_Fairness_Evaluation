import logging
import pickle
from collections import defaultdict
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import  mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from feature_selection_type import FeatureSelectionType
from helper_functions import (
    load_models1,
    load_models2,
    lower_upper_bound,
)
from ml_models import MLModels
from set_logger import set_log_config
from settings import (
    CSV_PATH,
    SELECTION_METHOD,
)

set_log_config()
logger = logging.getLogger()


class FeatureALgo:
    def __init__(self) -> None:
        self.selector = {
            "lasso": self.lasso,
            "mutual_information": self.mutual_information,
        }

    def permutation_import(self, X, y, model, cohort_name):
        perm_fet = permutation_importance(estimator=model, X=X, y=y, n_repeats=50)
        feature = X.columns
        feature_importance = perm_fet.importances_mean
        pd.DataFrame({"features": feature, "scores": feature_importance}).to_csv(
            f"{CSV_PATH}/20260216/features_with_score/{cohort_name}_permutation_features_new.csv"
        )

    def shap(self, X, y, model, cohort_name):
        explainer = shap.Explainer(model)
        res = explainer(X)
        shap_df = pd.DataFrame(np.mean(res.values, axis=2), columns=res.feature_names)
        final_shap = shap_df.mean()
        pd.DataFrame(
            {"features": final_shap.keys(), "scores": final_shap.values}
        ).to_csv(f"{CSV_PATH}/20260216/features_with_score/{cohort_name}_shap_features_new.csv")

    def lasso(
        self, X: pd.DataFrame, y: pd.DataFrame, coef=0.00001
    ) -> List[str]:
        
        lasso_features = defaultdict(list)
        lasso = Lasso()
        prev_param_grid = {"alpha": [0.01, 0.001, 0.0001]}
        X1 = X
        X = StandardScaler().fit_transform(X)
        for iter_number in range(100):
            # logger.info(f"number of iteration for lasso: {iter_number}")
            scorer = make_scorer(mean_squared_error)
            lasso_gridsearch = GridSearchCV(lasso, prev_param_grid, scoring=scorer, cv=5)
            lasso_gridsearch.fit(X, y)
            best_alpha = lasso_gridsearch.best_estimator_.alpha
            prev_param_grid = {"alpha": lower_upper_bound(best_alpha)}
            features = X1
            features_score = lasso_gridsearch.best_estimator_.coef_
            for f, v in zip(features, features_score):
                lasso_features[f].append(v)
        pd.DataFrame({"alpha", best_alpha}).to_csv("20260216_best_alpha.csv")
        for key, value in lasso_features.items():
            lasso_features[key] = np.mean(value)
        pd.DataFrame(
            {"features": lasso_features.keys(), "scores": lasso_features.values()}
        ).to_csv(f"{CSV_PATH}/20260216/features_with_score/{cohort_name}_lasso_features_new.csv")


    def select_features_models1(
        self,
        features: pd.DataFrame,
        label: pd.DataFrame,
        feature_selection_type: str,
        ml_model_train,
        cohort_name,
    ) -> List[str]:
        rfe_features = defaultdict()
        feature_number = len(features.columns)

        feature_selector = load_models1(
            feature_selection_type=feature_selection_type,
            mlmodel=ml_model_train,
            X=features,
            y=label,
            feature_number=feature_number,
        )
        if feature_selector.estimator:
            fitted_features = feature_selector.fit(features, label)
            try:
                final_features = fitted_features.get_feature_names_out()
                final_features_score = feature_selector.estimator_.feature_importances_
            except:
                final_features = fitted_features.k_feature_names_

            for f, v in zip(final_features, final_features_score):
                rfe_features[f] = v
            pd.DataFrame(
                {"features": rfe_features.keys(), "scores": rfe_features.values()}
            ).to_csv(
                f"{CSV_PATH}/20260216/features_with_score/{cohort_name}_{feature_selection_type}_features_new.csv"
            )


    def selec_feature_models2(
        self,
        features: pd.DataFrame,
        label: pd.DataFrame,
        feature_selection_type: str,
        ml_model_train,
        cohort_name,
    ) -> List[str]:
        max_iter = len(features.columns)
        featute_selector = load_models2(
            feature_selection_type=feature_selection_type,
            mlmodel=ml_model_train,
            X=features,
            y=label,
            max_iter=max_iter,
        )
        try:
            fitted_features = featute_selector.fit(features, label)
        except:
            fitted_features = featute_selector.fit(np.array(features), np.array(label))
        try:
            final_features_score = fitted_features.estimator_.feature_importances_
            final_features = fitted_features.get_feature_names_out()
        except:
            try:
                final_features_score = fitted_features.estimator.feature_importances_
                final_features = fitted_features.get_feature_names_out()
            except:
                final_features = features.columns[fitted_features.support_]
                final_features_score = fitted_features.estimator.feature_importances_
        pd.DataFrame(
            {"features": fitted_features.estimator.feature_names_in_, "scores": final_features_score}
        ).to_csv(
            f"{CSV_PATH}/20260216/features_with_score/{cohort_name}_{feature_selection_type}_features_new.csv"
        )


    def mutual_information(
        self, X: pd.DataFrame, y: pd.DataFrame, cohort_name
    ) -> pd.DataFrame:
        mi_result = mutual_info_classif(X, y, random_state=42)
        features_list = X.columns.to_list()
        final_scores = pd.DataFrame({"features": features_list, "scores": mi_result})
        final_scores.to_csv(
            f"{CSV_PATH}/20260216/features_with_score/{cohort_name}_mutual_info_features_new.csv"
        )



def select_features(
    X,
    y,
    method,
    ind,
    ml_model_train,
    cohort_name,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_methods = FeatureALgo()
    if method in [
        FeatureSelectionType.BorutaPy.name,
        FeatureSelectionType.SelectBySingleFeaturePerformance.name,
        FeatureSelectionType.SelectByShuffling.name,
        FeatureSelectionType.GeneticSelectionCV.name,
    ]:
        select_features = all_methods.selec_feature_models2(
            features=X,
            label=y,
            feature_selection_type=method,
            ml_model_train=ml_model_train,
            cohort_name=cohort_name,
        )

    elif method in [
        FeatureSelectionType.RFE.name,
        FeatureSelectionType.SelectFromModel.name,
        FeatureSelectionType.SequentialFeatureSelector.name,
    ]:
        select_features = all_methods.select_features_models1(
            features=X,
            label=y,
            feature_selection_type=method,
            ml_model_train=ml_model_train,
            cohort_name=cohort_name,
        )
    elif method == "lime":
        all_methods.select_lime(X, y, ml_model_train, cohort_name)
    elif method == "permute":
        all_methods.permutation_import(X, y, ml_model_train, cohort_name)
    elif method == "shap":
        all_methods.shap(X, y, ml_model_train, cohort_name)
    else:
        selector_method = all_methods.selector.get(method)


def change_label(data):
    real_data["AD_diagnosis"] = np.select([(real_data["CDX_Cog"] == 0) & (real_data["01_AB_FBB_Global_SUVR"] < 1.08),
                                           (real_data["CDX_Cog"] == 1) & (real_data["01_AB_FBB_Global_SUVR"] >1.08),
                                           (real_data["CDX_Cog"] == 2) & (real_data["01_AB_FBB_Global_SUVR"] >1.08),
                                           ],[0,1, 2],default=np.nan
    )
    real_data.dropna(subset='AD_diagnosis', inplace=True)
    real_data.drop(["Unnamed: 0", "CDX_Cog", "01_AB_FBB_Global_SUVR", "01_AB_FBB_pos"], axis=1, inplace=True)
    return real_data

if __name__ == "__main__":
    import os
    from pathlib import Path
    from helper_functions import test_individual

    real_data = pd.read_csv(f"{CSV_PATH}/final_preprocessed_HABS_new.csv")
    real_data = change_label(real_data)
    label = "AD_diagnosis"
    y = real_data[label]
    file_name = real_data["file_name"]
    X = real_data.drop(["file_name", label], axis=1)
    cohort_name = "combined"
    for ind, methodes in enumerate(SELECTION_METHOD):
        if f"{cohort_name}_{label}_{methodes}_RF_new_hbs.pkl" not in os.listdir(
            CSV_PATH / "20260216"/ "model"
        ):
            logger.info(f"{methodes}, {cohort_name}")
            if methodes == FeatureSelectionType.BorutaPy.name:
                ml_model_train = MLModels(
                    model="XGB", X=X, y=y
                ).train_classifier()
                pickle.dump(
                    ml_model_train,
                    open(
                        f"{CSV_PATH}/20260216/model/{cohort_name}_{label}_{methodes}_XGB_new_hbs.pkl",
                        "wb",
                    ),
                )
            else:
                ml_model_train = MLModels(
                    model="RF", X=X, y=y
                ).train_classifier()
                pickle.dump(
                    ml_model_train,
                    open(
                        f"{CSV_PATH}/20260216/model/{cohort_name}_{label}_{methodes}_RF_new_hbs.pkl",
                        "wb",
                    ),
                )
                logger.info(f"{methodes}, {cohort_name}")
                select_features(
                X, y, methodes, ind, ml_model_train, cohort_name
                )
        else:
            if f"{cohort_name}_{methodes}_features_new.csv" not in os.listdir(CSV_PATH / "20260216"/ "features_with_score"): 
                ml_model_train = pickle.load(open( f"{CSV_PATH}/20260216/model/{cohort_name}_{label}_{methodes}_RF_new_hbs.pkl", "rb"))
                logger.info(f"{methodes}, {cohort_name}")
                select_features(
                X, y, methodes, ind, ml_model_train, cohort_name
                )
            else:
                logger.info(f"All good for {cohort_name}_{label}_{methodes}")
