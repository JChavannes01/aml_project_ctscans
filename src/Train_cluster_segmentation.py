import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import gzip
import pickle
from segmentation_models import get_unet_128

version = 1
do_training = True

# Directory where the preprocessed data is stored.
input_dir = "/deepstore/datasets/course/aml/group4/All_Data"
output_dir = "/deepstore/datasets/course/aml/group4/models"

def train_segmentation():
    # Load segmentation
    lab_pattern = re.compile(r'labels-(\d+).npy')
    labels_files = filter(lambda f: re.match(
        lab_pattern, f), os.listdir(input_dir))
    labels = np.array([])
    segmentation = np.array([])
    images = np.array([])
    for f in labels_files:
        # Load label so slices with no liver can be removed
        labels = np.append(labels, np.load(os.path.join(input_dir, f)))

        # Load segmentation (stack all frames vertically), datatype of segmentation array is uint8
        # Loading segmentation this way ensures that labels and images have the same order
        file_num = re.match(lab_pattern, f).group(1)
        with gzip.GzipFile(os.path.join(input_dir, 'liver_segmentation-{}.npy.gz'.format(file_num)), "r") as file:
            if segmentation.size == 0:
                segmentation = np.load(file)
            else:
                segmentation = np.concatenate((segmentation, np.load(file)),axis=0) # (Z, X, Y)

        # Load images (stack all frames vertically)
        with gzip.GzipFile(os.path.join(input_dir, 'images-{}.npy.gz'.format(file_num)), "r") as file:
            if images.size == 0:
                images = np.load(file)
            else:
                images = np.concatenate((images, np.load(file)),axis=0) # (Z, X, Y)

    # Filter out all slices containing no liver part
    images = images[labels == 1,:,:]
    segmentation = segmentation[labels == 1,:,:]

    # Add new dimension (explicit mention that we have only one color channel)
    images = images[:,:,:,np.newaxis]
    segmentation = segmentation[:,:,:,np.newaxis]

    # Change range from 0-255 to 0-1 (datatype change from uint8 to float64)
    images = images / 255.0

    # Change datatype labels from float64 to booleans
    labels = labels.astype(bool, copy=False)

    # Split data in train, validation and test set
    images_train, images_test, segmentation_train, segmentation_test, indices_train, indices_test = train_test_split(images,segmentation,range(images.shape[0]),test_size=0.2,random_state=42)
    images_train, images_validation, segmentation_train, segmentation_validation, indices_train, indices_validation = train_test_split(images_train, segmentation_train, indices_train, test_size=0.2,random_state=42)
    
    # Create model
    model = get_unet_128((128, 128, 1), 1)

    # Start training
    history = model.fit(images_train, segmentation_train, batch_size=32, epochs=12, validation_data=(images_validation,segmentation_validation))

    # Determine dice coefficient on train, validation and test data
    dice_coeff_train = model.evaluate(images_train, segmentation_train)
    dice_coeff_validation = model.evaluate(images_validation, segmentation_validation)
    dice_coeff_test = model.evaluate(images_test, segmentation_test)
    dice_coeff = {"train": dice_coeff_train, "validation": dice_coeff_validation, "test": dice_coeff_test}

    # Determine confusion matrix for test data
    predictions = model.predict(images_test)
    predictions_list = predictions >= 0.5

    con_matrix = confusion_matrix(segmentation_test.flatten(), predictions_list.flatten())

    # Save model and all data
    model_path = os.path.join(output_dir, "unet_128_v{}.h5".format(version))
    model.save(model_path)
    pickle_path = os.path.join(output_dir, "unet_128_v{}.pkl".format(version))
    with open(pickle_path, "wb") as f:
        pickle.dump([indices_train, indices_test, indices_validation, history.history, dice_coeff, con_matrix, predictions], f)

def main():
    # Checks to make sure we dont accidentally override our previous models.
    if os.path.exists("/deepstore/datasets/course/aml/group4/models/unet_128_v{}-exists".format(version)):
        raise ValueError("Version {} already exists, aborting training...".format(version))
    
    # Add a placeholder file to indicate that this version has already been trained:
    with open("/deepstore/datasets/course/aml/group4/models/unet_128_v{}-exists".format(version), 'w') as f:
        pass

    if do_training:
        train_segmentation()

if __name__ == "__main__":
    main()