import os
import numpy as np
import nibabel as nib
from nilearn import plotting
#import cv2

def get_slope_inter(img):
    # Return the slope and y-intercept of the data.
    return img.dataobj.slope, img.dataobj.inter

# Testing file
file_path = r"data\Training_Batch1\volume-1.nii"

# load the header to check what the linear transformation should be.
img = nib.load(file_path)

scale, offset = get_slope_inter(img)
print(get_slope_inter(img))

data = img.get_fdata()

# Show 3d viewer of all the slices in the CT scan.
nib.viewers.OrthoSlicer3D(img.get_fdata(), img.affine, title='slicer').show()

print(data.shape)

# Do linear transformation to get the original data.
# y = mx + b => x = (y-b)/m
pic = (data[:,:,0]-offset) / scale

# Filter out non-interesting CT data
pic[pic < -250] = -250
pic[pic > 250] = 250
pic = np.round(255 * ( pic + 250 ) / 500).astype('uint8', copy=False)
print(np.min(pic), np.max(pic), np.std(pic), np.mean(pic), pic.shape)

'''
# Show a single slice with opencv
while True:
    cv2.imshow('testing MRI', pic)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
'''