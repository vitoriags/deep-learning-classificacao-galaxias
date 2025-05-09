# src/config.py

from pathlib import Path

# Caminho base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Diretórios principais
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

IMAGES_DIR = DATA_DIR / "images"
ORIGINALS_DIR = IMAGES_DIR / "originals"
ZIP_PATH_ORIGINALS = DATA_DIR / "originals.zip"
TEST_CSV_PATH = DATA_DIR / "test_table.csv"

# ResNet152V2: 224 || InceptionResNetV2: 299 || Inception v3: 299 || Alexnet: 227 || MobileNetV2: 424 || AttentionResNet56: 224 || AttentionResNet92: 224 || GoogLenet: 224 || CoAtNet4: 224
IMG_RESOLUTION = 227
