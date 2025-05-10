import os
from glob import glob

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

import config as config
import funcoes as funcoes
import models as models

# Extract images from the ZIP file if not already extracted
funcoes.extrair_zip(config.ZIP_PATH_SEGMENTED, config.IMAGES_DIR)

# Load the test CSV file into a DataFrame
testdf = pd.read_csv(config.TEST_CSV_PATH, sep=",")

# Select columns
colunas = ["images", "dr7objid", "gz2_class", "asset_id"]
testdf = testdf[colunas]
print(testdf.head(30))

# Convert class labels to integers
RotuloImagensTest = list(testdf["gz2_class"].map(lambda nome: int(nome)))
print(RotuloImagensTest)

# Print class distribution
funcoes.contagemimagens(RotuloImagensTest, "segmented")

# Define directories for training, validation, and test sets
train_directory = config.DATA_DIR / "images" / "segmented" / "train"
validation_directory = config.DATA_DIR / "images" / "segmented" / "validation"
test_directory = config.DATA_DIR / "images" / "segmented" / "test"

# Load datasets using custom split function
train_ds = funcoes.train_test_validation_split(train_directory, config.IMG_RESOLUTION)
val_ds = funcoes.train_test_validation_split(
    validation_directory, config.IMG_RESOLUTION
)
test_ds = funcoes.train_test_validation_split(test_directory, config.IMG_RESOLUTION)

# Initialize the model
model = models.Alexnet()

# Print model architecture
print(model.summary())

# Ensure the models directory exists
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define checkpoint path
filepath = config.MODELS_DIR / "model_test_alexnet"

# Skip training if model already exists
if os.path.exists(filepath):
    print("Model already exists. Skipping training.")
else:
    # Define callback to save the best model based on validation accuracy
    callbacks = ModelCheckpoint(
        filepath=str(filepath),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        save_freq="epoch",
        options=None,
    )

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model for 1 epoch
    model.fit(
        train_ds,
        epochs=40,
        validation_data=val_ds,
        validation_freq=1,
        callbacks=callbacks,
    )

    # Predict on the test set
    pred = model.predict(test_ds, batch_size=5, verbose=1)
    pred[:10]

    # Convert predictions to class indices
    predicao = np.argmax(pred, axis=-1)

    # Print first 10 predictions
    for x in predicao[:10]:
        print(x)

    # Print evaluation metrics
    print(metrics.confusion_matrix(RotuloImagensTest, predicao))
    print(metrics.classification_report(RotuloImagensTest, predicao))
    print(metrics.roc_auc_score(RotuloImagensTest, predicao))
    print(metrics.accuracy_score(RotuloImagensTest, predicao))
    print(metrics.f1_score(RotuloImagensTest, predicao))
    print(metrics.recall_score(RotuloImagensTest, predicao))
