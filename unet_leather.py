# -*- coding:utf-8 -*-
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
from data_leather import *


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        # print(imgs_train)
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self, pretrained_weights=None):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape) 

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape) 
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape) 

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        #merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        merge6 = concatenate([drop4, up6], axis=3)
        print(up6)
        print(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        print(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        #merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        merge7 = concatenate([conv3, up7], axis=3)
        print(up7)
        print(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        print(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        #merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        #merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        merge9 = concatenate([conv1, up9], axis=3)
        print(up9)
        print(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        print(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print(conv9)
        conv9 = Conv2D(12, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv9 shape:", conv9.shape)

        conv10 = Conv2D(12, 1, activation='softmax')(conv9)
        print(conv10)
        model = Model(inputs, conv10)
        print(model.summary())
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('unet_leather.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=100, verbose=1,
                  validation_split=0.1, shuffle=True, callbacks=[model_checkpoint]) #
        #model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=1,
        #          shuffle=True, callbacks=[model_checkpoint])  #
        #model.fit_generator(imgs_train, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/leather_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('./results/leather_mask_test.npy')
        piclist = []
        for line in open("./results/leather.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        print('imgs.shape[0]',imgs.shape[0]) # 圖片數量
        print('imgs.shape[1]',imgs.shape[1]) # width
        print('imgs.shape[2]',imgs.shape[2]) # height
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)

            for k in range(len(img)):
                for j in range(len(img[k])):  # cv2.imwrite也是BGR顺序
                    num = np.argmax(imgs[i][k][j])
                    if num == 0:
                        img[k][j] = [128, 128, 128]
                    elif num == 1:
                        img[k][j] = [128, 0, 0]
                    elif num == 2:
                        img[k][j] = [192, 192, 128]
                    elif num == 3:
                        img[k][j] = [255, 69, 0]
                    elif num == 4:
                        img[k][j] = [128, 64, 128]
                    elif num == 5:
                        img[k][j] = [60, 40, 222]
                    elif num == 6:
                        img[k][j] = [128, 128, 0]
                    elif num == 7:
                        img[k][j] = [192, 128, 128]
                    elif num == 8:
                        img[k][j] = [64, 64, 128]
                    elif num == 9:
                        img[k][j] = [64, 0, 128]
                    elif num == 10:
                        img[k][j] = [64, 64, 0]
                    elif num == 11:
                        img[k][j] = [0, 128, 192]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            #print(path)
            cv2.imwrite(path, img)


if __name__ == '__main__':
    H5 = 'unet_leather.hdf5'
    myunet = myUnet()
    model = myunet.get_unet(pretrained_weights=H5)
    # model.summary()
    # plot_model(model, to_file='model.png')
    myunet.train()
    myunet.save_img()
