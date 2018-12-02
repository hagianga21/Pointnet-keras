import numpy as np
import os
import sys
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
#from schedules import onetenth_50_75
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool2D, Activation

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

numPoints = 2048
numCategories = 40

def mat_mul(A, B):
    return tf.matmul(A, B)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def show_point_clouds(pointCloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = pointCloud[:, 0]
    ys = pointCloud[:, 1]
    zs = pointCloud[:, 2]
    ax.scatter(xs, ys, zs, c='r', marker='o')
    plt.show()
#show_point_clouds(train_points_r[0])

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

#DATA PREPARATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#Download data if doesnot exist
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

TRAIN_FILES = getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

#LOAD TRAIN POINTS
train_points = None
train_labels = None
for fn in range(len(TRAIN_FILES)):
    print('----Train file' + str(fn) + '-----')
    current_data, current_label = loadDataFile(TRAIN_FILES[fn])
    cur_points = current_data.reshape(1, -1, 3)
    cur_labels = current_label.reshape(1, -1)
    print(cur_points.shape)
    if train_labels is None or train_points is None:
        train_labels = cur_labels
        train_points = cur_points
    else:
        train_labels = np.hstack((train_labels, cur_labels))
        train_points = np.hstack((train_points, cur_points))
train_points_r = train_points.reshape(-1, numPoints, 3)
train_labels_r = train_labels.reshape(-1, 1)

#LOAD TEST POINTS
test_points = None
test_labels = None
for fn in range(len(TEST_FILES)):
    print('----Test file' + str(fn) + '-----')
    current_data, current_label = loadDataFile(TEST_FILES[fn])
    cur_points = cur_points.reshape(1, -1, 3)
    cur_labels = cur_labels.reshape(1, -1)
    if test_labels is None or test_points is None:
        test_labels = cur_labels
        test_points = cur_points
    else:
        test_labels = np.hstack((test_labels, cur_labels))
        test_points = np.hstack((test_points, cur_points))
test_points_r = test_points.reshape(-1, numPoints, 3)
test_labels_r = test_labels.reshape(-1, 1)

Y_train = np_utils.to_categorical(train_labels_r, numCategories)
Y_test = np_utils.to_categorical(test_labels_r, numCategories)

input_points = Input(shape=(numPoints, 3))
#PointNet Architecture
# input_Transformation_net
T_Net1 = Conv1D(64, 1, activation= 'relu')(input_points)
T_Net1 = BatchNormalization()(T_Net1)
T_Net1 = Conv1D(128, 1, activation= 'relu')(T_Net1)
T_Net1 = BatchNormalization()(T_Net1)
T_Net1 = Conv1D(1024, 1, activation= 'relu')(T_Net1)
T_Net1 = BatchNormalization()(T_Net1)
T_Net1 = MaxPooling1D(pool_size=2048)(T_Net1)

T_Net1 = Dense(512, activation='relu')(T_Net1)
T_Net1 = BatchNormalization()(T_Net1)
T_Net1 = Dense(256, activation='relu')(T_Net1)
T_Net1 = BatchNormalization()(T_Net1)
T_Net1 = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(T_Net1)
T_Net1 = Reshape((3, 3))(T_Net1)

#matrix multiply
inputTransform = Lambda(mat_mul, arguments={'B': T_Net1})(input_points)

#forward net
g = Conv1D(64, 1, activation='relu')(inputTransform)
g = BatchNormalization()(g)
g = Conv1D(64, 1, activation='relu')(g)
g = BatchNormalization()(g)

#feature transform net
T_Net2 = Conv1D(64, 1, activation='relu')(g)
T_Net2 = BatchNormalization()(T_Net2)
T_Net2 = Conv1D(128, 1, activation='relu')(T_Net2)
T_Net2 = BatchNormalization()(T_Net2)
T_Net2 = Conv1D(1024, 1, activation='relu')(T_Net2)
T_Net2 = BatchNormalization()(T_Net2)
T_Net2 = MaxPooling1D(pool_size=2048)(T_Net2)

T_Net2 = Dense(512, activation='relu')(T_Net2)
T_Net2 = BatchNormalization()(T_Net2)
T_Net2 = Dense(256, activation='relu')(T_Net2)
T_Net2 = BatchNormalization()(T_Net2)
T_Net2 = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(T_Net2)
T_Net2 = Reshape((64, 64))(T_Net2)

#matrix multiply
featureTransform = Lambda(mat_mul, arguments={'B': T_Net2})(g)

#forward net
g = Conv1D(64, 1, activation='relu')(featureTransform)
g = BatchNormalization()(g)
g = Conv1D(128, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Conv1D(1024, 1, activation='relu')(g)
g = BatchNormalization()(g)

#global feature
global_feature = MaxPooling1D(pool_size=2048)(g)

c = Dense(512, activation='relu')(global_feature)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(256, activation='relu')(c)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(numCategories, activation='softmax')(c)
prediction = Flatten()(c)

model = Model(inputs=input_points, outputs=prediction)
print(model.summary())

lr = 0.001
adam = Adam(lr=lr)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if not os.path.exists('./Clsresults/'):
    os.mkdir('./Clsresults/')

callbacks = [ReduceLROnPlateau(monitor='loss',
                               factor=0.1,
                               patience=2,
                               verbose=1,
                               min_lr=0.000001,
                               mode='min'),
             ModelCheckpoint(monitor='val_acc',save_weights_only=True,
                             filepath='./Clsresults/pointnet.h5',
                             save_best_only=True,
                             mode='max', verbose=1) ,
            ]

#TRAINING DATA
# Fit model on training data
for i in range(1,100):
    # rotate and jitter the points
    train_points_rotate = rotate_point_cloud(train_points_r)
    train_points_jitter = jitter_point_cloud(train_points_rotate)
    validation_data = (test_points_r, Y_test)
    history = model.fit(train_points_jitter, Y_train, 
                        batch_size=32, 
                        epochs=1, 
                        shuffle=True, 
                        verbose=1, 
                        validation_data = validation_data, 
                        callbacks=callbacks)
    #model.fit(train_points_jitter, train_labels_r, batch_size=32, epochs=1, shuffle=True, verbose=1)
    s = "Current epoch is:" + str(i)
    print(s)
    if i % 5 == 0:
        score = model.evaluate(test_points_r, Y_test, verbose=1)
        #score = model.evaluate(test_points_r, test_labels_r, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

# score the model
score = model.evaluate(test_points_r, Y_test, verbose=1)
#score = model.evaluate(test_points_r, test_labels_r, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])