import logging
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

path = Path(__file__).parent.parent


class MLModels:
    def __init__(self, model, X, y) -> None:
        self.model = model
        self.X = X
        self.y = y

    def ml_model_factory(self):
        models = {
            "LR": self.LR,
            "RF": self.RF,
            "XGB": self.XGB,
            "SVM": self.SVM,
            "SGD": self.SGD,
            "LGBM": self.LGBM,
        }
        return models.get(self.model)()

    def train_classifier(self) -> Tuple:
        """
        Train a classifier

        """
        # try:
        classfier = self.ml_model_factory()
        return classfier.fit(self.X, self.y).best_estimator_

        # except (TypeError, ValueError):
        #     return None

    def LGBM(self):
        param_dists = {
            "n_estimators": [int(x) for x in np.linspace(start=80, stop=140, num=20)],
            "colsample_bytree": [0.7, 0.8],
            "max_depth": [50, 90, 100],
            "num_leaves": [20, 30, 40],
            "reg_alpha": [1.1, 1.2, 1.3],
            "reg_lambda": [1.1, 1.2, 1.3],
            "min_split_gain": [0.3, 0.4],
            "subsample": [0.7, 0.8, 0.9],
            "subsample_freq": [3],
            "boosting_type": ["gbdt", "dart"],
            "learning_rate": [0.005, 0.01],
        }
        return GridSearchCV(
            lgb.LGBMClassifier(),
            param_dists,
            return_train_score=True,
            cv=5,
            scoring="f1_macro",
        )

    def RF(self):
        n_estimators = [80, 100, 150, 200, 250, 300]
        max_depth = [20, 60, 80, 100]
        min_samples_split = [35, 50, 70, 90]
        min_samples_leaf = [20, 50, 70]
        bootstrap = [True, False]
        criterion = ["gini", "entropy"]
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
            "criterion": criterion,
        }
        return GridSearchCV(
            RandomForestClassifier(),
            params,
            return_train_score=True,
            cv=5,
            scoring="f1_macro",
        )

    def SGD(self):
        param_dists = {
            "loss": ["squared_hinge", "hinge"],
            "alpha": [0.01, 0.001, 0.0001],
            "epsilon": [0.01, 0.001],
        }

        return GridSearchCV(
            SGDClassifier(),
            param_dists,
            return_train_score=True,
            cv=5,
            scoring="f1_macro",
        )

    def XGB(self):
        class_number = len(self.y.unique())
        if class_number == 2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"


    # "loss" : ['log_loss', 'deviance', 'exponential'] = "log_loss",
    # "learning_rate" :  0.1,
    # "n_estimators" :  100,
    # "subsample" :  1,
    # "criterion" : ['friedman_mse', 'squared_error'] = "friedman_mse",
    # "min_samples_split" :  2,
    # "min_samples_leaf" :  1,
    # "min_weight_fraction_leaf" : 0,
    # "max_depth" :  3,
    # "min_impurity_decrease" : 0,
    # "max_features" : ['auto', 'sqrt', 'log2'],
    # "verbose" :  0,
    # "max_leaf_nodes" : Int,
    # "warm_start" :  False,
    # "validation_fraction" : 0.1,
    # "tol" : 0.0001,
    # "ccp_alpha" : 0
        
        # xgc = GradientBoostingClassifier()
            # nthread=1, objective=objective, n_jobs=1, num_class=class_number)
        # xgb_grid = {
        #     "loss":["log_loss"],
        #     "min_child_weight": [1],
        #     "gamma": [0.5],
        #     "learning_rate": [0.03],
        #     "max_depth": [2],
        #     "n_estimators": [1],
        # }
        xgb_grid = {
            "loss":['log_loss', 'exponential'],
            "subsample": [0.7],
            "learning_rate": [ 0.001, 0.0001],
            # "criterion": ['friedman_mse', 'squared_error'],
            "n_estimators": [100, 150, 300],
            "min_samples_split" :  [15, 20, 50],
            "max_depth": [50, 90, 100],
            "max_features": ['auto', 'log2'],
        }
        return GridSearchCV(
            GradientBoostingClassifier(),
            xgb_grid,
            return_train_score=True,
            cv=5,
            scoring="f1_macro",
        )

    def SVM(self):
        params = {
            "C": np.logspace(-6, 6, 5),
            "gamma": np.logspace(-8, 8, 4),
            "kernel": ["rbf", "linear", "sigmoid"],
            "class_weight": ["balanced"],
            "tol": np.logspace(-4, -1, 30),
            "random_state": [1],
        }

        return GridSearchCV(
            SVC(),
            params,
            return_train_score=True,
            cv=5,
            scoring="f1_macro",
        )

    def LR(self):
        LRmodel = LogisticRegression(max_iter=300)
        logic_grid = {
            "solver": ["liblinear", "sag", "saga"],
            "penalty": ["l1", "l2"],  # , "elasticnet"],
            "class_weight": ["balanced"],
            "C": [1, 10, 100],
        }
        return GridSearchCV(
            LRmodel,
            logic_grid,
            return_train_score=True,
            cv=5,
        )
