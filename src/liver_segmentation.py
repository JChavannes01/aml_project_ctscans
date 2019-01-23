import os
import re
import gzip
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from segmentation_models import get_unet_128

# Directory of the pickle file
output_dir = r"models\models trained with new split"
# Name of pickle file
pkl_file = r"unet_128_v101.pkl"

# Old function (not usefull anymore)
def load_segmentation():
    model = get_unet_128((128, 128, 1), 1)
    #model.load_weights(r"models/liver_segmentation.h5")
    
    with open(os.path.join(output_dir, pkl_file), 'rb') as f:
        myHistory, dice_coeff, con_matrix, predictions = pickle.load(f)
    
    # Print dice coefficient saved in pickle file
    print(f"Dice coefficient of best saved model: {dice_coeff}")
    
    # Show dice coefficient on train and validation data
    hor_axis = 1+np.arange(len(myHistory[b'train_dice_coeff']))
    max_indice = np.argmax(myHistory[b'val_dice_coeff'])
    plt.figure(figsize=[8,6])
    plt.plot(hor_axis, myHistory[b'train_dice_coeff'],'r',linewidth=3.0)
    plt.plot(hor_axis, myHistory[b'val_dice_coeff'],'b',linewidth=3.0)
    plt.plot(hor_axis[max_indice],dice_coeff[b"test"][1],'k+',markersize=14.0)
    plt.legend(['Training Dice Coefficient', 'Validation Dice Coefficient', 'Test Dice Coefficient'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Dice Coefficient',fontsize=16)
    plt.title('Dice Coefficient Curves',fontsize=16)
    plt.grid()
    plt.show()
    
    '''
    # Show images from the test set with openCV
    i = 0
    while True:
        predictions_t = model.predict(images_test[i][np.newaxis])
        #predictions_t = predictions[i][np.newaxis]
        combined2 = np.zeros((128,128,3))
        combined2[:,:,0] = images_test[i,:,:,0] #Blue
        combined2[:,:,1] = segmentation_test[i,:,:,0]/2 #Green
        combined2[:,:,2] = predictions_t[0,:,:,0]/2 # Red
        dice = np.round(model.evaluate(images_test[i][np.newaxis],segmentation_test[i][np.newaxis])[1],2)
        cv2.imshow(f'image, i: {i}, dice_coeff: {dice}', cv2.resize(images_test[i], dsize=(512, 512)))
        #cv2.imshow(f'prediction, i: {i}, accuracy: {acc}', cv2.resize(predictions_t[0]*255, dsize=(512, 512)))
        cv2.imshow(f'prediction, i: {i}, dice_coeff: {dice}', cv2.resize(predictions_t[0]*255., dsize=(512, 512)))
        cv2.imshow(f'segmentation, i: {i},  dice_coeff: {dice}', cv2.resize(segmentation_test[i]*255., dsize=(512, 512)))
        #combined = cv2.addWeighted(images_test[i],0.5,predictions_t[0],0.5,0)
        #cv2.imshow(f'combined, i: {i}, dice_coeff: {dice}', cv2.resize(combined, dsize=(512, 512)))
        cv2.imshow(f'combined2, i: {i}, dice_coeff: {dice}', cv2.resize(combined2, dsize=(512, 512)))
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(0) & 0xFF == ord('d'):
            if i < images_test.shape[0]-1:
                i = i + 1
                cv2.destroyAllWindows()
        if cv2.waitKey(0) & 0xFF == ord('a'):
            if i > 0:
                i = i - 1
                cv2.destroyAllWindows()
    '''

def main():
    load_segmentation()

if __name__ == "__main__":
    main()