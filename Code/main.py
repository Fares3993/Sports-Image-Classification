import cv2
import keras
import numpy as np
import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.xception import preprocess_input
from random import shuffle
from tqdm import tqdm
import tensorflow as tf 
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.estimator import regression
# import keras
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense#vGG16 with keras
# from tensorflow.keras import layers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.preprocessing import *
# from tflearn.layers.core import input_data, dropout, fully_connected
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras.layers.normalization import batch_normalization
from keras.models import *
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, Concatenate, Average,Maximum
from keras.layers import Input
from keras.utils.data_utils import get_file
import csv
import pandas as pd


TRAIN_DIR = 'C:/Users/HYPER/Desktop/ProjectNN/Code/NN Dataset/Train'
TEST_DIR = 'C:/Users/HYPER/Desktop/ProjectNN/Code/NN Dataset/Test'
IMG_SIZE = 224
bool_model = True # true --> VGG16 , false --> AlexNet

#     """ Create an one-hot encoded vector from image name """
def create_label(image_name):
    word_label = image_name.split('_')
    if word_label[0] == 'Basketball':
        return np.array([1,0, 0, 0, 0,0])
    elif word_label[0] == 'Football':
        return np.array([0,1, 0, 0, 0,0])
    elif word_label[0] == 'Rowing':
        return np.array([0,0, 1, 0, 0,0])
    elif word_label[0] == 'Swimming':
        return np.array([0,0, 0, 1, 0,0])
    elif word_label[0] == 'Tennis':
        return np.array([0,0, 0, 0, 1,0])
    elif word_label[0] == 'Yoga':
        return np.array([0,0, 0, 0, 0,1])


# preprocessing
def zoom_at(img, zoom=1, angle=0, coord=None):
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def Rotation(img, deg):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        label = create_label(img)

        # Augmentation
        flip1 = cv2.flip(img_data, 1)
        flip2 = cv2.flip(img_data, 0)
        flip3 = cv2.flip(img_data, -1)
        rotated = Rotation(img_data, 45)
        flip4 = cv2.flip(flip3, 1)
        flip5 = cv2.flip(rotated, 0)
        flip6 = cv2.flip(flip5, -1)
        rotated1 = Rotation(flip6, 90)
        zoom = cv2.flip(rotated1, -1)
        zoom1 = cv2.flip(flip6, 1)

        training_data.append([np.array(img_data), label])
        training_data.append([np.array(flip1), label])
        training_data.append([np.array(flip2), label])
        training_data.append([np.array(flip3), label])
        training_data.append([np.array(rotated), label])
        training_data.append([np.array(flip4), label])
        training_data.append([np.array(flip5), label])
        training_data.append([np.array(flip6), label])
        training_data.append([np.array(rotated1), label])
        training_data.append([np.array(zoom), label])
        training_data.append([np.array(zoom1), label])

    shuffle(training_data)
    # np.save('train_data.npy', training_data)
    return training_data
# if (os.path.exists('C:\Users\HYPER\Desktop\ProjectNN\Code\train_data.npy')): # If you have already created the dataset:
#     train =np.load('C:\Users\HYPER\Desktop\ProjectNN\Code\train_data.npy',allow_pickle=True)
# else:#If dataset is not created:

train = create_train_data()

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_train = np.asarray([i[1] for i in train])

if(bool_model):
    #VGG16 LAST MODEL
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',WEIGHTS_PATH_NO_TOP)

    model = Sequential()
    # input_layer=input_data(shape=(224,224,3))
    input_layer=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    conv_l1=Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(input_layer)
    conv_l2=Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(conv_l1)
    pool1=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_l2)
    conv_l3=Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
    conv_l4=Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv_l3)
    pool2=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_l4)
    conv_l4=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
    conv_l5=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv_l4)
    conv_l6=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv_l5)
    pool3=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_l6)
    conv_l6=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
    conv_l7=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_l6)
    conv_l8=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_l7)
    pool4=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_l8)
    conv_l8=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
    conv_l9=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_l8)
    conv_l10=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv_l9)
    pool5=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_l10)
    model=Model(inputs=input_layer,outputs=pool5)
    model.load_weights(weights_path)
    out=model.output
    model.trainable=False
    LAYER=tf.keras.layers.GlobalAveragePooling2D()
    out=LAYER(out)
    out=Dense(3072,activation="relu")(out)
    out=Dense(6,activation="softmax")(out)
    Vgg16=Model(inputs=model.input,outputs=out)

    if (os.path.exists('C:/Users/HYPER\Desktop/ProjectNN/Models/Last_VGG 16/Vgg16 .tfl')):
        Vgg16 = tf.keras.models.load_model('C:/Users/HYPER\Desktop/ProjectNN/Models/Last_VGG 16/Vgg16 .tfl')
        print("\n\nVGG16 Loaded Successfully\n")
    else:
        optimization = tf.keras.optimizers.SGD(lr=.0001, decay=1e-6, momentum=0.9, nesterov=True, name="SGD")
        Vgg16.compile(optimizer=optimization, loss='categorical_crossentropy', metrics=['accuracy'])
        Vgg16.fit(X_train, Y_train, validation_split=0.2, batch_size=1, epochs=8, verbose=1)
        Vgg16.save('Vgg16.tfl')

else:
    #Alex Net
    AlexNet = Sequential()

    # 1st Convolutional Layer
    #  image_shape=(IMG_SIZE,IMG_SIZE,3)
    AlexNet.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    AlexNet.add(Activation('relu'))
    # Max Pooling
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    AlexNet.add(Activation('relu'))
    # Max Pooling
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    AlexNet.add(Activation('relu'))

    # 4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    AlexNet.add(Activation('relu'))

    # 5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    AlexNet.add(Activation('relu'))
    # Max Pooling
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    #model.load_wights(wieghts_path)
    #model.trainable=false

    # Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, input_shape=(224,224,)))
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    AlexNet.add(Dense(4096))
    AlexNet.add(Activation('relu'))
    # Add Dropout
    AlexNet.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    AlexNet.add(Dense(1000))
    AlexNet.add(Activation('relu'))
    # Add Dropout
    AlexNet.add(Dropout(0.4))

    # Output Layer
    AlexNet.add(Dense(6))
    AlexNet.add(Activation('softmax'))

    AlexNet.summary()


    if (os.path.exists('C:/Users/HYPER\Desktop/ProjectNN/Models/AlexNet/Alex_net.tfl')):
        AlexNet =tf.keras.models.load_model('C:/Users/HYPER\Desktop/ProjectNN/Models/AlexNet/Alex_net.tfl')
        print("\n\nAlexNet Loaded Successfully\n")

    else:
        optimization = tf.keras.optimizers.SGD(lr=.0001, decay=1e-6, momentum=0.9, nesterov=True, name="SGD")
        AlexNet.compile(optimizer=optimization, loss='categorical_crossentropy', metrics=['accuracy'])
        AlexNet.fit(X_train, Y_train,validation_split=0.2 ,batch_size=1, epochs=13, verbose=1)
        AlexNet.save('Alex_net.tfl')





#create testing CSV
idx =[]
img_name = []
for test_img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, test_img)
    img_data = cv2.imread(path)
    img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
    img_data = img_data.reshape(-1,IMG_SIZE, IMG_SIZE,3)

    if(bool_model):
        prediction = Vgg16.predict([img_data])[0]
    else:
        prediction = AlexNet.predict([img_data])[0]
    idx.append(np.argmax(prediction))
    img_name.append(test_img)
out = pd.DataFrame()
out["image_name"]=img_name
out["label"]=idx
if(bool_model):
    out.to_csv('VGG16 out.csv',index=False)
else:
    out.to_csv('AlexNet out.csv', index=False)
print(out.head())

######################################################################################################################
# # lab model
# tf.compat.v1.reset_default_graph()
# conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# conv1 = conv_2d(conv_input, 32, 5, activation ='relu',padding='same')
# pool1 = max_pool_2d(conv1, 5)

# conv2 = conv_2d(pool1, 64, 5, activation='relu',padding='same')
# pool2 = max_pool_2d(conv2, 5)

# conv3 = conv_2d(pool2, 128, 5, activation='relu',padding='same')
# pool3 = max_pool_2d(conv3, 5)

# conv4 = conv_2d(pool3, 128, 5, activation='relu',padding='same')
# pool4 = max_pool_2d(conv4, 5)

# conv5 = conv_2d(pool4, 32, 5, activation='relu',padding='same')
# pool5 = max_pool_2d(conv5, 5)

# fully_layer = fully_connected(pool5, 512, activation='relu')
# fully_layer1 = fully_connected(fully_layer, 512, activation='relu')

# cnn_layers = fully_connected(fully_layer1, 6, activation='softmax')

# cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)


# #fares & Mohammes
# # epochs = 15
# # VGG model ver 16
# tf.compat.v1.reset_default_graph()
# conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# conv1 = conv_2d(conv_input, 64, 3, activation='relu')
# conv2 = conv_2d(conv1, 64, 3, activation='relu')
# pool1 = max_pool_2d(conv2, 3)

# conv3 = conv_2d(pool1, 128, 3, activation='relu')
# conv4 = conv_2d(conv3, 128, 3, activation='relu')
# pool2 = max_pool_2d(conv4, 3)

# conv5 = conv_2d(pool2, 256, 3, activation='relu')
# conv6 = conv_2d(conv5, 256, 3, activation='relu')
# pool3 = max_pool_2d(conv6, 3)

# conv7 = conv_2d(pool3, 512, 3, activation='relu')
# conv8 = conv_2d(conv7, 512, 3, activation='relu')
# conv9 = conv_2d(conv8, 512, 3, activation='relu')
# pool4 = max_pool_2d(conv9, 3)

# conv10 = conv_2d(pool4, 512, 3, activation='relu')
# conv11 = conv_2d(conv10, 512, 3, activation='relu')
# conv12 = conv_2d(conv11, 512, 3, activation='relu')
# pool5 = max_pool_2d(conv12, 3)



# fully_layer = fully_connected(pool4, 4096, activation='relu')
# fully_layer = dropout(fully_layer, 0.5)
# fully_layer1 = fully_connected(fully_layer, 4096, activation='relu')
# fully_layer1 = dropout(fully_layer1, 0.5)
# fully_layer2 = fully_connected(fully_layer1, 1000, activation='relu')
# fully_layer2 = dropout(fully_layer2, 0.5)

# cnn_layers = fully_connected(fully_layer2, 6, activation='softmax')

# cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
# print (X_train.shape)

# # # #Khaled & Mohammed
# tf.compat.v1.reset_default_graph()
# conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# conv1 = conv_2d(conv_input, 32, 5, activation='relu')
# pool1 = max_pool_2d(conv1, 2)

# conv2 = conv_2d(pool1, 32, 5, activation='relu')
# pool2 = max_pool_2d(conv2, 2)

# conv3 = conv_2d(pool2, 64, 3, activation='relu')
# pool3 = max_pool_2d(conv3, 3)
# conv4 = conv_2d(pool3, 64, 3, activation='relu')


# fully_layer = fully_connected(conv4, 256, activation='relu')
# fully_layer1 = dropout(fully_layer, 0.2)

# cnn_layers = fully_connected(fully_layer1, 6, activation='softmax')

# cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
# print (X_train.shape)

# # updated lab model
# tf.compat.v1.reset_default_graph()
# conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

# conv1 = conv_2d(conv_input, 32, 5, activation='relu')
# pool1 = max_pool_2d(conv1, 5)

# conv2 = conv_2d(pool1, 64, 5, activation='relu')
# pool2 = max_pool_2d(conv2, 5)

# conv3 = conv_2d(pool2, 128, 5, activation='relu')
# pool3 = max_pool_2d(conv3, 5)

# conv4 = conv_2d(pool3, 64, 5, activation='relu')
# pool4 = max_pool_2d(conv4, 5)

# conv5 = conv_2d(pool4, 32, 5, activation='relu')
# pool5 = max_pool_2d(conv5, 5)

# conv6 = conv_2d(pool5, 64, 5, activation='relu')
# pool6 = max_pool_2d(conv6, 5)

# conv7 = conv_2d(pool6, 128, 5, activation='relu')
# pool7 = max_pool_2d(conv7, 5)

# conv8 = conv_2d(pool7, 64, 5, activation='relu')
# pool8 = max_pool_2d(conv8, 5)

# conv9 = conv_2d(pool8, 32, 5, activation='relu')
# pool9 = max_pool_2d(conv9, 5)

# fully_layer = fully_connected(pool9, 1024, activation='relu')
# fully_layer = dropout(fully_layer, 0.5)

# cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

# cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

# #AlexNet_model by using tflearn
# tf.compat.v1.reset_default_graph()
# Input_layer=conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# Conv_layer1=conv_2d(Input_layer,96,11,activation='relu')
# Max_pool_layer1=max_pool_2d(Conv_layer1, 3)
# Conv_layer2=conv_2d(Max_pool_layer1,256,5,activation='relu')
# Max_pool_layer2=max_pool_2d(Conv_layer2, 3)
# Conv_layer3=conv_2d(Max_pool_layer2,384,3,activation='relu')
# Conv_layer4=conv_2d(Conv_layer3,384,3,activation='relu')
# Conv_layer5=conv_2d(Conv_layer4,256,3,activation='relu')
# Max_pool_layer3=max_pool_2d(Conv_layer5, 3)

# fully_conn1=fully_connected(Max_pool_layer3, 4096, activation='relu')
# fully_conn1 = dropout(fully_conn1, 0.5)
# fully_conn2=fully_connected(fully_conn1, 4096, activation='relu')
# fully_conn2= dropout(fully_conn2, 0.5)

# fully_conn2_softmax=fully_connected(fully_conn2, 6, activation='softmax')
# CNN_model=regression(fully_conn2_softmax,optimizer='SGD', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model=tflearn.DNN(CNN_model, tensorboard_dir='log', tensorboard_verbose=3)
# print(X_train.shape)

#ZFNet_Model
# tf.compat.v1.reset_default_graph()
# Input_layer=conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# Conv_layer1=conv_2d(Input_layer,32,7,activation='relu')
# Max_pool_layer1=max_pool_2d(Conv_layer1, 3)
# Conv_layer2=conv_2d(Max_pool_layer1,52,5,activation='relu')
# Max_pool_layer2=max_pool_2d(Conv_layer2, 3)
# Conv_layer3=conv_2d(Max_pool_layer2,512,3,activation='relu')
# Conv_layer4=conv_2d(Conv_layer3,1024,3,activation='relu')
# Conv_layer5=conv_2d(Conv_layer4,512,3,activation='relu')
# Max_pool_layer3=max_pool_2d(Conv_layer5, 3)
# fully_conn1=fully_connected(Max_pool_layer3, 128, activation='relu')
# fully_conn2=fully_connected(fully_conn1, 32, activation='relu')
# fully_conn2_softmax=fully_connected(fully_conn2, 6, activation='softmax')
# CNN_model=regression(fully_conn2_softmax,optimizer='SGD', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model=tflearn.DNN(CNN_model, tensorboard_dir='log', tensorboard_verbose=3)
# print(X_train.shape)

# print(X_train[0].shape)

# AlexNet with Keras
# #Instantiate an empty model
# image_shape=(IMG_SIZE,IMG_SIZE,3)
# model_AlexNet = Sequential()
# #first_convoltion_layer
# model_AlexNet.add(Conv2D(filters=96,input_shape=image_shape,kernel_size=(11,11),strides=(4,4),padding='valid'))
# model_AlexNet.add(Activation('relu'))
# # Max Pooling1
# model_AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# #second_convolution_layer
# model_AlexNet.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
# model_AlexNet.add(Activation('relu'))
# # Max Pooling2
# model_AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# #third_convolution_layer
# model_AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
# model_AlexNet.add(Activation('relu'))
# #fourth_convolution_layer
# model_AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
# model_AlexNet.add(Activation('relu'))
# #fifth_convolution_layer
# model_AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
# model_AlexNet.add(Activation('relu'))
# # Max Pooling3
# model_AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# # Passing it to a Fully Connected layer
# model_AlexNet.add(Flatten())
# #first_fully_connectedLayer
# model_AlexNet.add(Dense(4096, input_shape=image_shape))
# model_AlexNet.add(Activation('relu'))
# # Adding Dropout to prevent overfitting
# model_AlexNet.add(Dropout(0.4))
# #second_fully_connectedLayer
# model_AlexNet.add(Dense(4096))
# model_AlexNet.add(Activation('relu'))
# # Adding Dropout to prevent overfitting
# model_AlexNet.add(Dropout(0.4))
# #output_layer
# model_AlexNet.add(Dense(6))
# model_AlexNet.add(Activation('softmax'))

# model_AlexNet.summary()
# #compiling and running model
# model_AlexNet.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

# # if (os.path.exists('model.tfl.meta')):
# #     model.load('/kaggle/working/model.tfl')
# # else:
# X_train,X_test,Y_train,Y_test = train_test_split(X_train, Y_train, test_size=0.20, train_size=0.80, shuffle=True)
# model.fit({'input': X_train}, {'targets': Y_train}, n_epoch = 25,
#           validation_set=({'input': X_test}, {'targets': Y_test}),
#       snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
# model.save('model.tfl')