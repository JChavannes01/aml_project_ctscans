import os
import re
import gzip
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cv2
import pickle
import matplotlib.pyplot as plt


# Directory of the pickle file
output_dir = r"models\models trained with new split"
# Name of pickle file
pkl_file = r"CNN_v101.pkl"

def CheckPrediction(Prediction,label):
    if label >= 0.5 and Prediction[1] >= 0.5:
        return True
    elif label < 0.5 and Prediction[1] < 0.5:
        return True
    else:
        return False

def load_classifier():    
    with open(os.path.join(output_dir, pkl_file), 'rb') as f:
        myHistory, accuracy, con_matrix = pickle.load(f,encoding='bytes')
    
    # Print accuracy saved in pickle file 
    print(f"Accuracy of best saved model: {accuracy}")

    # Print confusion matrix
    print(f"Confusion matrix of best saved model (for test data): {con_matrix}")

    # Show accuracy on train and validation data
    hor_axis = 1+np.arange(len(myHistory[b'train_acc']))
    max_indice = np.argmax(myHistory[b'val_acc'])
    plt.figure(figsize=[8,6])
    plt.plot(hor_axis, myHistory[b'train_acc'],'r',linewidth=3.0)
    plt.plot(hor_axis, myHistory[b'val_acc'],'b',linewidth=3.0)
    plt.plot(hor_axis[max_indice],accuracy[b"test"][1],'k+',markersize=14.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.grid()
    plt.show()

    '''
    # Can be used to show images
    # Show the testing image with opencv
    j = 0
    i = Wrong_classified[j]
    while True:
        cv2.imshow(f'i: {i}, Pred.: {np.round(predictions[i],2)}, Lab.: {np.round(labels_test[i])}, Result: {CheckPrediction(predictions[i],labels_test[i])}', cv2.resize(images_test[i], dsize=(512, 512)))

        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(0) & 0xFF == ord('d'):
            if j < Wrong_classified.size-1:#i <labels_test.size-1:
                j = j + 1
                i = Wrong_classified[j]
                cv2.destroyAllWindows()
        if cv2.waitKey(0) & 0xFF == ord('a'):
            if j > 0:
                j = j - 1
                i = Wrong_classified[j]
                cv2.destroyAllWindows()
    '''
def main():
    load_classifier()

if __name__ == "__main__":
    main()