# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:22:37 2019

@author: XiaoKunJianKe
"""
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import mode
from PIL import Image
import matplotlib.pyplot as plt
img_dir='E:/Mnnist_32_32/train_label_9/9_1001.jpg'
label_dir='E:/Mnnist_32_32/train_image_9/9_1001.jpg'
logs_train_dir='E:/Mnist_32_32/log/'
label_origin=Image.open(label_dir)
label_origin=np.array(label_origin)
image = Image.open(img_dir)
image=np.array(image)
image=image/255
image=np.reshape(image,[32,32])

with tf.Graph().as_default():
    image_=tf.cast(image,tf.float32)
    image_=tf.reshape(image_,[1,32,32,1])
    yyy=mode.train(image_)
    x = tf.placeholder(tf.float32, shape = [32,32])
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init) 
        print('下载模型')
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.all_model_checkpoint_paths:            
            global_step = ckpt.all_model_checkpoint_paths[4].split('/')[-1].split('.')[-1]
            saver.restore(sess, ckpt.all_model_checkpoint_paths[4]) # 重载模型
            print('Loading success, global_step is %s' % global_step)
        else:
            print("No checkpoint file found")
        prediction = sess.run(yyy, feed_dict = {x: image})
        pre=prediction
        pre=np.reshape(pre,[32,32])
        prediction=sess.run(tf.nn.sigmoid(prediction))
        prediction=np.reshape(prediction,[32,32])
        rows,cols=pre.shape
        for i in range(rows):
            for j in range(cols):
                if pre[i,j]<=0.5:
                   pre[i,j]=0
                else:
                   pre[i,j]=1
        
      

plt.subplot(141)
plt.imshow(image,cmap='gray')
plt.title('Fresnel diffraction_image')
plt.subplot(142)
plt.imshow(prediction,cmap='gray')
plt.title('reconstruction_label')
plt.subplot(143)
plt.imshow(label_origin,cmap='gray')
plt.title('origin_label')
plt.subplot(144)
plt.imshow(pre,cmap='gray')
plt.title('binarization_label',pad=-1200)
plt.show()    



