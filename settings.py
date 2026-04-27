import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


BASE_PATH = Path(__file__).parent.parent / "data"
CSV_PATH = BASE_PATH / "csv_data"
HTML_PATH = BASE_PATH / "html_data"
PNG_PATH = BASE_PATH / "png_data"
ONEDRIVE_LOCSTION = os.environ.get("ONEDRIVE_LOCSTION")
PLOT = False
SINGLE = True
USER_ID = "PTID"
COL_1 = "SUMMARY_SUVR"
COL_2 = "AMYLOID_STATUS"
Z_SCORE = "PHC_AB42"
ADNI_ADSP = BASE_PATH / "adni_adsp"

W2V_MODEL = os.environ.get("W2V_MODEL")
FASTTEXT = os.environ.get("FASTTEXT")
GROUPBY_DATASET = bool(os.environ.get("GROUPBY_DATASET"))
LOGGING_LEVEL = "INFO"
IMPUTER_NAME_SUBSET = os.environ.get("IMPUTER_NAME_SUBSET")
IMPUTER_NAME_WHOLE = os.environ.get("IMPUTER_NAME_WHOLE")
FORMAT = "%(asctime)s.%(msecs)03d %(name)-8s %(levelname)-4s %(message)s"
DATE_FORMAT = "%m-%d %H:%M:%S"
COUNTER = 5

SELECTION_METHOD = [
    "lasso",
    "RFE",
    "SelectFromModel",
    "mutual_information",
    # "SelectByShuffling",
    # "permute",
    "SelectBySingleFeaturePerformance", 
    # "SequentialFeatureSelector"
    # "BorutaPy",
    "shap"
]
    # "shap",
    # "pearson",
    # "GeneticSelectionCV",
    # "BorutaPy",
# ]
CL_LABEL = ["AD_diagnosis"]
ML_MODELS = [["BorutaPy"], ["lasso"], SELECTION_METHOD]

FEATURE_TO_DROP = ["Med_ID", "Visit_ID", "CDR_Sum"]
SPECIAL_VALUE = os.environ.get("SPECIAL_VALUE")
CLASS_NAME = os.environ.get("CLASS_NAME")
COMBINED_COHORT = os.environ.get("COMBINED_COHORT")
X_ITER = 50
ALPHA = 0.05
POS_NEG_THR = 0.0
NUMBER_FEATURES = [5, 10, 15, 20]
MODELS_B = ["RF", "LGBM", "XGB"]
# MODELS = ["XGB", "LR", "RF", "SVM", "SGD", "LGBM"]
LIST_OF_COLUMNS_CON_NONCON = ["Visit_ID", "CDR_Sum"]
DIAGNOSIS_LABEL = ["01_AB_FBB_Global_SUVR", "CDR_Sum", "Visit_ID", "Med_ID"]
INDEX_OF_INTEREST = 1
CUT_OFF = 1.08
# GROUP_ID = ["Med_ID", "file_name_1"]
GROUP_ID = "PTID"
GROUPBY_COLUMNS = ["AD_diagnosis", "file_name"]
# GROUPBY_COLUMNS = ["CDR_Diagnosis"]
GROUPBY_COLUMNS_STAT = "Visit_ID"
COLUMNS = [
    "ID_Gender",
    "Interview_Site",
    "ID_MaritalStatus",
    "ID_Education_Degree",
    "HealthStatus",
    "AUDIT_1",
    "AUDIT_2",
    "AUDIT_3",
    "Smoke_Currently",
    # "r2_simoa_plasma_Ab40",
    # "r2_simoa_plasma_Ab42",
    # "r2_simoa_plasma_Total_Tau",
    # "r2_simoa_plasma_NfL",
    # "r3_simoa_plasma_Ab40",
    # "r3_simoa_plasma_Ab42",
    # "r3_simoa_plasma_Total_Tau",
    # "r3_simoa_plasma_pTau181",
    # "r3_simoa_plasma_NfL",
    "CDR_Sum",
]
COLUMNS_FOR_GROUPING = [
    "ID_Gender",
    "ID_MaritalStatus",
    "ID_Education_Degree",
    "HealthStatus",
    "file_name",
]
FEATURE_LIST = [
    "Clinical Dementia Rating Scale",
    "ADAS-Cog11" "ADAS-Cog13",
    "Mini-Mental State Examination",
    "RAVLT immediate",
    "RAVLT learning",
    "RAVLT forgetting",
    "RAVLT forgetting percent",
    "Functional Activities Questionnaire",
    "Montreal Cognitive Assessment",
    "Ventricles",
    "Hippocampus",
    "Whole brain volume",
    "Entorhinal cortical volume",
    "Fusiform cortical volume",
    "Middle temporal cortical volume",
    "Intracranial volume",
    "Florbetapir- PET",
    "Fluorodeoxyglucose (FDG) - PET",
    "Beta-amyloid (CSF)",
    "Total tau",
    "Phosphorylated tau",
    "Diagnosis",
]
