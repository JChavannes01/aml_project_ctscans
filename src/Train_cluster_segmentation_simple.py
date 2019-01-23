import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from segmentation_models import *
import tensorflow as tf
from util import OUTPUT_DIR, load_test_val_train_files_segmentation

version = 1
do_training = True
model_id = "unet_128"

def train_segmentation():
    # Load segmentation
    images_train, images_validation, images_test, segmentation_train, segmentation_validation, segmentation_test = load_test_val_train_files_segmentation(version)
    
    # Create model
    model = get_unet_128((128, 128, 1), 1)

        # Define callbacks
        model_path = os.path.join(OUTPUT_DIR, "{}_v{}.h5".format(model_id, version))
        callbacks = [tf.keras.callbacks.ModelCheckpoint(monitor='val_acc',
                filepath=model_path,
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                period=1),
                accuracyHistory]

    # Start training
    history = model.fit(images_train, segmentation_train, batch_size=32, epochs=18, validation_data=(images_validation,segmentation_validation), callbacks=callbacks)

    # Add validation dice coefficient to accuracyHistory class
    accuracyHistory.add_validation_accuracy(history.history)

    # Determine dice coefficient on train, validation and test data
    model.load_weights(model_path)
    dice_coeff_train = model.evaluate(images_train, segmentation_train)
    dice_coeff_validation = model.evaluate(images_validation, segmentation_validation)
    dice_coeff_test = model.evaluate(images_test, segmentation_test)
    dice_coeff = {"train": dice_coeff_train, "validation": dice_coeff_validation, "test": dice_coeff_test}

    # Determine confusion matrix for test data
    predictions = model.predict(images_test)
    predictions_list = predictions >= 0.5

    con_matrix = confusion_matrix(segmentation_test.flatten(), predictions_list.flatten())

    pickle_path = os.path.join(OUTPUT_DIR, "{}_v{}.pkl".format(model_id, version))
    with open(pickle_path, "wb") as f:
        pickle.dump([accuracyHistory.myHistory, dice_coeff, con_matrix, predictions], f)

def main():
    # Checks to make sure we dont accidentally override our previous models.
    if os.path.exists(os.path.join(OUTPUT_DIR, "{}_v{}-exists".format(model_id, version))):
        raise ValueError("Version {} already exists, aborting training...".format(version))
    
    # Add a placeholder file to indicate that this version has already been trained:
    with open(os.path.join(OUTPUT_DIR, "{}_v{}-exists".format(model_id, version)), 'w') as f:
        pass

    if do_training:
        train_segmentation()

if __name__ == "__main__":
    main()