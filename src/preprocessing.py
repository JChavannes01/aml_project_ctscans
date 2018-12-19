import os
import numpy as np
import nibabel as nib
import re
import gzip
import cv2

# Directory where the .nii files are located.
train_data_dir = r"data\Training_Batch1"

# Directory where the preprocessed data will be stored.
output_dir = r"data\Training_Batch1\preprocessed"


def preprocess_labels():
    # Converts all segmentation .nii files to .npy files which contain only the labels for each layer.
    # Where the labels are: {0: No liver, 1: Liver}
    segmentation_files = filter(lambda f: re.match(
        seg_pattern, f), os.listdir(train_data_dir))
    seg_pattern = re.compile(r'segmentation-(\d+).nii')
    for f in segmentation_files:
        print(f'Extracting labels from {f}')
        nii_file = nib.load(os.path.join(train_data_dir, f))
        data = nii_file.get_fdata()

        labels = np.zeros((data.shape[2]))
        for layer in range(data.shape[2]):
            if np.sum(data[:, :, layer] == 1) > 0:
                labels[layer] = 1

        file_num = re.match(seg_pattern, f)[1]
        save_filepath = os.path.join(output_dir, f'labels-{file_num}.npy')
        np.save(save_filepath, labels)
        del nii_file, data, labels


def preprocess_images():
    # Converts all .nii files to .npy.gz (compressed numpy) files which contain only the images (layer, x, y) = [0-255]
    vol_pattern = re.compile(r'volume-(\d+).nii')
    volume_files = filter(lambda f: re.match(
        vol_pattern, f), os.listdir(train_data_dir))
    window = (0, 255)  # start, size
    output_dimension = (128, 128)  # desired output size of each layer

    for i, f in enumerate(volume_files):
        if i > 0:
            break
        print(f'Extracting images from {f}')
        nii_file = nib.load(os.path.join(train_data_dir, f))
        data = nii_file.get_fdata()

        images = np.zeros(
            (data.shape[-1], *output_dimension), dtype='uint8')  # (Z, X, Y)
        for layer in range(data.shape[2]):
            pic = data[:, :, layer]

            # Filter out the non interesting areas
            pic[pic < window[0]] = window[0]
            pic[pic > sum(window)] = sum(window)  # start + size
            pic = np.round(
                255 * (pic - window[0]) / window[1]).astype('uint8', copy=False)

            # Resize to the desired output dimension
            pic = cv2.resize(pic, dsize=output_dimension)
            images[layer] = pic

        file_num = re.match(vol_pattern, f)[1]
        save_filepath = os.path.join(
            output_dir, f'images-{file_num}.npy.gz')
        with gzip.GzipFile(save_filepath, 'w') as gzipfile:
            np.save(gzipfile, images)
        del nii_file, data, images


def load_test_labels():
    labels = np.load(os.path.join(output_dir, 'labels-0.npy'))
    print(labels.shape)


def load_test_images():
    # Use gzip to load images since the files are compressed
    with gzip.GzipFile(os.path.join(output_dir, 'images-0.npy.gz'), "r") as f:
        images = np.load(f)

    # Show the testing image with opencv
    while True:
        cv2.imshow('testing MRI', cv2.resize(images[60], dsize=(512, 512)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main():
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    preprocess_labels()
    # preprocess_images()
    load_test_labels()
    # load_test_images()


if __name__ == "__main__":
    main()
