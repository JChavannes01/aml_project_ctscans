import os
import numpy as np
import nibabel as nib
import re
import gzip
import cv2

# Directory where the .nii files are located.
train_data_dir = r"data\Training_Batch1"

# Directory where the preprocessed data will be stored.
output_dir = r"data\Training_Batch1\preprocessed_transfer_learning"

def gen_images():
    # Converts all slices of the .nii files to images and places them in the correct subdirectories for they category
    # E.g. a slice which contains liver cells, will be placed in the folder output_dir/liver

    output_dimension = (128, 128)  # desired output size of each layer
    window = (0., 255.)  # start, size. bounds for the pixel value window
    
    seg_pattern = re.compile(r'segmentation-(\d+).nii')
    
    file_nums = [re.match(seg_pattern, fp)[1] for fp in os.listdir(train_data_dir) if re.match(seg_pattern, fp)]
    file_nums.sort(key=int)
    
    for num in file_nums:
        print(f'Processing {num}/{file_nums[-1]}')
        # load Volume.nii file
        vol_nii_file = nib.load(os.path.join(train_data_dir, f'volume-{num}.nii'))
        vol_data = vol_nii_file.get_fdata()

        # load Segmentation.nii file
        seg_nii_file = nib.load(os.path.join(train_data_dir, f'segmentation-{num}.nii'))
        seg_data = seg_nii_file.get_fdata()

        # Iterate over all the slices in the scan
        for layer in range(seg_data.shape[2]):
            # Determine the class label
            label = 'no_liver' if np.sum(seg_data[:, :, layer] == 1) == 0 else 'liver'

            # Extract, transform and save the image.
            pic = vol_data[:, :, layer]

            # Filter out the non interesting areas
            pic[pic < window[0]] = window[0]
            pic[pic > sum(window)] = sum(window)  # start + size
            pic = (255*(pic - window[0]) / window[1]).astype('uint8', copy=False)

            # Resize to the desired output dimension
            pic = cv2.resize(pic, dsize=output_dimension)

            # Save the image to the correct folder:
            filename = f"img_{num}_{layer}.jpg"
            save_fp = os.path.join(output_dir, label, filename)
            #np.repeat(pic[..., np.newaxis], 3, -1)
            cv2.imwrite(save_fp, pic, [cv2.IMWRITE_JPEG_QUALITY, 95]) # Stack pic 3 times to fake RGB from grayscale
            del pic

        del vol_nii_file, vol_data, seg_nii_file, seg_data


def load_test_labels():
    labels = np.load(os.path.join(output_dir, 'labels-28.npy'))
    print(labels.shape)


def load_test_images():
    # Use gzip to load images since the files are compressed
    with gzip.GzipFile(os.path.join(output_dir, 'images-0.npy.gz'), "r") as f:
        images = np.load(f)

    # Show the testing image with opencv
    while True:
        cv2.imshow('testing MRI', cv2.resize(images[72], dsize=(512, 512)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main():
    # Make sure that the paths for the ouput dir and the label subdirectories exist.
    for path in [output_dir, os.path.join(output_dir, "no_liver"), os.path.join(output_dir, "liver")]:
        if not os.path.isdir(path):
            os.mkdir(path)

    gen_images()


if __name__ == "__main__":
    main()
