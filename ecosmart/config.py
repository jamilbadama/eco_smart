import os
from dotenv import load_dotenv

load_dotenv()

# Paths
# Default to current directory + daic_woz_data if not specified in env
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(os.getcwd(), "daic_woz_data"))
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")
TEST_SPLIT = os.path.join(DATA_ROOT, "test_split_Depression_AVEC2017.csv")

# Feature Dimensions
AUDIO_DIM = 74 # COVAREP
MEL_DIM = 80 # Mel-spectrogram
VIDEO_DIM = 711 # CLNF
TEXT_DIM = 768 # DistilBERT

# Hyperparameters
BATCH_SIZE = 8
MAX_SEQ_LEN = 500
LEARNING_RATE = 1e-4
EPOCHS = 100
SEED = 42
USE_RAW_AUDIO = True
FUSION_TYPE = 'early'
