from common import GENRES
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import rmsprop
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.layers import Input, Dense, Lambda, TimeDistributed, BatchNormalization, LSTM, Convolution2D,\
    MaxPooling2D, Bidirectional, Concatenate, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from optparse import OptionParser
import os

SEED = 42
BATCH_SIZE = 32
EPOCH_COUNT = 200
N_MEL= 256
INIT_CHANNEL = 1

def build_model(input_shape):
    input_layer = Input(input_shape, name='input')
    layer = input_layer
    layer = Convolution2D(filters=16, kernel_size=3, strides=1, padding='valid', data_format="channels_last", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2, strides=2)(layer)
    layer = Convolution2D(filters=32, kernel_size=3, strides=1, padding='valid', data_format="channels_last", activation='relu')(layer)
    layer = BatchNormalization(momentum=0.02)(layer)
    layer = MaxPooling2D(pool_size=2, strides=2)(layer)

    layer = Convolution2D(filters=64, kernel_size=3, strides=1, padding='valid', data_format="channels_last", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=2, strides=2)(layer)
    layer = Convolution2D(filters=32, kernel_size=3, strides=1, padding='valid', data_format="channels_last", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=4, strides=4)(layer)

    layerBi = input_layer
    layerBi = MaxPooling2D(pool_size=3, strides=3)(layerBi)
    layerBi = Lambda(lambda x: x[:, :, :, 0])(layerBi)
    layerBi = Bidirectional(LSTM(16, return_sequences=True))(layerBi)
    layerBi = TimeDistributed(Dense(len(GENRES)))(layerBi)
    time_distributed_merge_layer = Lambda(
        function=lambda x: K.mean(x, axis=1),
        output_shape=lambda shape: (shape[0],) + shape[2:],
        name='output_merged'
    )
    layerBi = time_distributed_merge_layer(layerBi)
    layerBi = Dense(64)(layerBi)

    layer = Flatten()(layer)
    layer = Dense(256, activation='relu')(layer)

    layer = Concatenate()([layer, layerBi])
    layer = Dense(10, activation='softmax')(layer)
    model_output = layer
    model = Model(input_layer, model_output)
    return model

def train_model(data, model_path):
    x = data['x']
    y = data['y']

    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.1,
            random_state=SEED)

    print('Building model...')
    x_train =np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)

    n_features = x_train.shape[2]
    row = x_train.shape[1]
    input_shape = (row, n_features, INIT_CHANNEL)

    model = build_model(input_shape)
    opt = rmsprop()
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

    print('Training...')
    tbCallBack = TensorBoard(log_dir='./Graph-final', histogram_freq=1, write_graph=True, write_images=True)
    model.fit(
        x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
        validation_data=(x_val, y_val), verbose=1, callbacks=[
            ModelCheckpoint(
                model_path, save_best_only=True, monitor='val_acc', verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=8, min_delta=0.01, min_lr=0.0000000015,
                verbose=1
            ),
            tbCallBack
        ]
    )

    return model

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--data_path', dest='data_path',
            default=os.path.join(os.path.dirname(__file__),
                'data/data.pkl'),
            help='path to the data pickle', metavar='DATA_PATH')
    parser.add_option('-m', '--model_path', dest='model_path',
            default=os.path.join(os.path.dirname(__file__),
                'models/model.h5'),
            help='path to the output model HDF5 file', metavar='MODEL_PATH')
    options, args = parser.parse_args()

    with open(options.data_path, 'rb') as f:
        data = pickle.load(f)

    train_model(data, options.model_path)
