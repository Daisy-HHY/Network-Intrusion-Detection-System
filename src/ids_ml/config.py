from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"

CATEGORICAL_COLUMNS = ["proto", "service", "state"]
BINARY_TARGET = "label"
MULTICLASS_TARGET = "attack_cat"
DROP_COLUMNS = ["id"]
RANDOM_STATE = 42
