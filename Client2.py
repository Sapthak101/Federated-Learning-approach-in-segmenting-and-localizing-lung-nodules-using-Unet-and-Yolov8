import flwr as fl
import numpy as np
import sys
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, TimeDistributed, Bidirectional, LSTM, GRU, Dense, Dropout, Input, concatenate
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Reshape
import os
import cv2
from keras.regularizers import l2
import argparse
import os
from catboost import CatBoostClassifier, Pool
import warnings 
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.pyplot as plt

import cv2
import os
import matplotlib.pyplot as plt

def read_images_and_masks(image_dir, mask_dir):
    images = []
    masks = []
    
    image_files = os.listdir(image_dir)
    
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        
        mask_filename = filename.split('.')[0] + '.jpg' 
        
        mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
        
        images.append(image)
        masks.append(mask)
        
    return images, masks

image_dir = 'Dataset_Processed/Client 2/images'
mask_dir = 'Dataset_Processed/Client 2/masks'

images, masks = read_images_and_masks(image_dir, mask_dir)

import cv2
import numpy as np

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    
    return image

def preprocess_data(images, masks, image_size, num_classes=None):
    preprocessed_images = []
    
    for image, mask in zip(images, masks):
        preprocessed_image = preprocess_image(image, image_size)
        preprocessed_images.append(preprocessed_image)
    
    return np.array(preprocessed_images), np.array(masks)

image_size = (256, 256)  
num_classes = 0  
preprocessed_images, preprocessed_masks = preprocess_data(images, masks, image_size, num_classes)

print(len(preprocessed_images))

import cv2

def resize_images(images):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_images.append(resized_image)
    return resized_images
resized_images = resize_images(preprocessed_masks)
preprocessed_masks=resized_images

preprocessed_masks=np.array(preprocessed_masks)
preprocessed_masks=preprocessed_masks.reshape(len(preprocessed_masks), 256,256,1)

import numpy as np
import cv2

def load_data(image_paths, mask_paths):
    images = []
    masks = []
    for x in range(len(image_paths)):
        img=image_paths[x]
        mask=mask_paths[x]
        mask = (mask > 0).astype(np.float32)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

preprocessed_images, preprocessed_masks = load_data(preprocessed_images, preprocessed_masks)

print("Image min/max values:", preprocessed_images.min(), preprocessed_images.max())
print("Mask min/max values:", preprocessed_masks.min(), preprocessed_masks.max())

train_img=preprocessed_images[0:600]
train_mask=preprocessed_masks[0:600]
val_img=preprocessed_images[601:800]
val_mask=preprocessed_masks[601:800]
test_img=preprocessed_images[801:949]
test_mask=preprocessed_masks[801:949]

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.losses import BinaryCrossentropy
from focal_loss import BinaryFocalLoss

def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', padding='same'):
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
    return conv
    
def unet(input_shape):
    inputs = Input(input_shape)
    
    # Down-sampling layers
    conv1 = conv_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 256)
    
    up5 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = conv_block(merge5, 128)
    
    up6 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = conv_block(merge6, 64)
    
    up7 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = conv_block(merge7, 32)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (256, 256, 3)
class_weights = {0: 1.0, 1: 100.0}
loss_function = BinaryFocalLoss(gamma=2)
model = unet(input_shape)
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
model.summary()

class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.bst = None
        self.config = None
        self.num=20
        self.index=0
        self.index1=0
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        global hist
        hist=[]

        model.set_weights(parameters)
        epochs = 90
        history = model.fit(train_img,train_mask,
                   steps_per_epoch = len(train_img),
                   batch_size = 32,
                   validation_split = 0.2,
                   validation_steps = len(val_img),
                   class_weight = class_weights,
                   callbacks=[
                               EarlyStopping(monitor = "val_loss",
                               patience = 5,
                               restore_best_weights = True), 
                               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, mode='min') 
                              ],
                   epochs = epochs)
        
        model_loss=pd.DataFrame(model.history.history)
        model_loss.plot()

        plt.xlabel('Number of Epochs')
        str1="Loss_accu_image_fed_round_client2"+str(self.index1)+".png"
        plt.savefig(str1)

        self.index1=self.index1+1
        return model.get_weights(), len(train_img), {}
    
    def evaluate(self, parameters, config):
        global y_pred
        model.set_weights(parameters)
        loss, accuracy=model.evaluate(test_img)
        print("Eval accuracy : ", accuracy)
        return loss, len(test_img), {"accuracy": accuracy}

fl.client.start_client(
    server_address="127.0.0.1:18080", 
    client=CifarClient().to_client())

from ultralytics import YOLO
model1=YOLO('yolov8n.pt')
results1 = model1.train(data="Dataset_Processed/Client 2/Yolo Format/detect.yaml", epochs=70)

pred_masks=model1.predict(test_img)
def postprocess_mask(mask):
    threshold = 0.5  
    mask = (mask > threshold).astype(np.uint8)
    return mask
predicted_mask = postprocess_mask(pred_masks)

i=0

num_images = 10
for i in range(num_images):
    plt.subplot(1, 2, 1)
    plt.imshow(test_img[i])
    plt.title('Image')
    plt.axis('off')
    plt.show()
    str1="Actual Image Client2"+str(i)+".png"
    plt.savefig(str1)

    plt.subplot(1, 2, 2)
    plt.imshow(test_mask[i], cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    plt.show()
    str1="Mask Image Client2"+str(i)+".png"
    plt.savefig(str1)

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask[i],cmap='gray')
    plt.title('Pred_Mask')
    plt.axis('off')    
    plt.show()
    str1="Predicted mask Client2"+str(i)+".png"
    plt.savefig(str1)    

    i=i+1

from skimage import io
from sklearn.metrics import confusion_matrix
import numpy as np

def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def compute_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    dice_coefficient = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice_coefficient

def compute_pixel_accuracy(y_true, y_pred):
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = y_true.size
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy

for x in range(10):
    iou_score = compute_iou(test_mask[x], predicted_mask[x])
    dice_coefficient = compute_dice(test_mask[x], predicted_mask[x])
    pixel_accuracy = compute_pixel_accuracy(test_mask[x], predicted_mask[x])

print("Intersection over Union (IoU):", iou_score)
print("Dice Coefficient:", dice_coefficient)
print("Pixel Accuracy:", pixel_accuracy)

test_img_sub=test_img[0:10]
results = model1(test_img) 
m=0

for result in results:
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs  
    result.show() 
    result.save(filename=m) 
    m=m+1