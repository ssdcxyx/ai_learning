    # -*- coding: utf-8 -*-
# @time       : 2/11/2019 9:17 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : InceptionV3.py
# @description:

from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Activation, GlobalAveragePooling2D, BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import matplotlib.pyplot as plt
import glob, os, cv2, random,time
import numpy as np

DATA_PATH = "./garbage"
HEIGHT, WIDTH = (384, 512)
NUM_CLASSES = 6
BATCH_SIZE = 8
NUM_EPOCHS = 100
MODEL_PATH = "./results/"
LOG_DIR = "./results/logs/"
TEST_DATA_PATH = "test.jpg"
# 冻结层
FREEZE_LAYERS = 11


def processing_data(data_path, validation_split=0.1):
    # -------------------------- 实现数据处理部分代码 ----------------------------
    train_data_generator = ImageDataGenerator(
        # 归一化
        # rescale=1./255,
        # 预处理
        preprocessing_function=preprocess_input,
        # 旋转角度
        rotation_range=40,
        # 裁剪强度
        shear_range=0.2,
        # 缩放强度
        zoom_range=0.2,
        # 水平偏移幅度
        width_shift_range=0.2,
        # 垂直偏移幅度
        height_shift_range=0.2,
        # 通道转换范围
        channel_shift_range=10,
        # 图片填充
        fill_mode="nearest",
        # 水平反转
        horizontal_flip=True,
        # 垂直反转
        vertical_flip=True,
        validation_split=validation_split
    )
    validation_data_generator = ImageDataGenerator(
            # rescale=1./255,
            preprocessing_function=preprocess_input,
            validation_split=validation_split)
    train_data = train_data_generator.flow_from_directory(
            data_path,
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            seed=1)
    validation_data = validation_data_generator.flow_from_directory(
            data_path,
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            seed=1)
    return train_data, validation_data


train_data, valid_data = processing_data(DATA_PATH)

labels = train_data.class_indices
print(labels)

labels = valid_data.class_indices
print(labels)


def model(train_data, valid_data, input_shape, save_model_path, log_dir):
    file_name = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(save_model_path, file_name), monitor='val_acc', verbose=1,
                                     save_best_only=True, save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5)

    pretrained_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

    x = pretrained_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    model = Model(pretrained_model.input, outputs)
    model.summary()

    # 冻结
    for layer in model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir)

    res = model.fit_generator(
        generator=train_data,
        epochs=NUM_EPOCHS,
        steps_per_epoch=train_data.samples // BATCH_SIZE,
        validation_data=valid_data,
        validation_steps=valid_data.samples // BATCH_SIZE,
        callbacks=[tensorboard, checkpoint, reduce_lr])
    model.save(save_model_path + "final.hdf5")
    return res, model


def plot_training_history(res):
    plt.plot(res.history['loss'], label='loss')
    plt.plot(res.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.plot(res.history['acc'], label='accuracy')
    plt.plot(res.history['val_acc'], label='val_accuracy')
    plt.legend()
    plt.show()


def predict(test_data_path, save_model_path):
    img = image.load_img(test_data_path)
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    model = load_model(save_model_path+"model_21-0.95.hdf5")
    _predict = model.predict(img)

    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    _predict = labels[np.argmax(_predict)]

    return _predict


train_data, valid_data = processing_data(DATA_PATH)
res, model = model(train_data, valid_data, (HEIGHT, WIDTH, 3), MODEL_PATH, LOG_DIR)
plot_training_history(res)
predict(TEST_DATA_PATH, MODEL_PATH)