# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:11:38 2019

@author: XiaoKunJianKe
"""
import tensorflow as tf
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
#train_image_dir1='E:/Mnist_32_32/train_image_'
#train_label_dir1='E:/Mnist_32_32/train_label_'
#logs_train_dir='E:/Mnist_32_32/log/'
def get_pair_batch(train_image_dir1,train_label_dir1,batch_size):
    image=[]
    label=[]
    images=[]
    labels=[]
    for i in range(10):        
        train_image_dir= train_image_dir1+str(i)+'/'
        train_label_dir=train_label_dir1+str(i)+'/'
        for file in os.listdir(train_image_dir):
            image.append(train_image_dir + file)
        for file in os.listdir(train_label_dir):
            label.append(train_label_dir+file)
    print('There are %d images \nThere are %d labels' %(len(image), len(label)))
    temp = np.array([image, label]) 
    temp = temp.transpose() #矩阵转置
    np.random.shuffle(temp) # 打乱存放的顺序      
    image_list = list(temp[:, 0]) # 获取图片
    label_list = list(temp[:, 1]) # 获取标签        
    for path in image_list:
        img=cv.imread(path,0)
        img=img/255.0
        images.append(img)
    for path in label_list:
        img=cv.imread(path,0)
        img=img/255.0
        labels.append(img)
       
    X=np.array(images,np.float32).reshape(-1,32,32,1)
    Y=np.array(labels,np.float32).reshape(-1,32,32,1)
    print('X的形状:',X.shape)
    print('Y的形状:',Y.shape)
    X=tf.cast(X, tf.float32)
    Y=tf.cast(Y, tf.float32)
    input_queue = tf.train.slice_input_producer([X, Y], shuffle=False)
 
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=5000)
    return  image_batch, label_batch 


def train(images):    
#第一层：卷积+RelU
    with tf.variable_scope('conv1') as scope:        
         weights=tf.get_variable('weights',shape=[3,3,1,32],dtype=tf.float32,initializer=
         tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
         biases=tf.get_variable('biases',shape=[32],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
         conv=tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')+biases
         conv1=tf.nn.relu(conv,name='conv1')


#第二层：卷积+RelU+max_pooling
    with tf.variable_scope('conv2') as scope:
        weights=tf.get_variable('weights',shape=[3,3,32,32],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[32],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv2=tf.nn.relu(conv,name='conv2')
        conv2=tf.nn.dropout(conv2,keep_prob=0.5,name='conv2_drop')
    with tf.variable_scope('max_pooling1') as scope:
        pool1=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pooling1')



#第三层：卷积+RelU
    with tf.variable_scope('conv3') as scope:
        weights=tf.get_variable('weights',shape=[3,3,32,64],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[64],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv3=tf.nn.relu(conv,name='conv3')



#第四层：卷积+RelU+max_pooling
    with tf.variable_scope('conv4') as scope:
        weights=tf.get_variable('weights',shape=[3,3,64,64],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[64],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(conv3,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv4=tf.nn.relu(conv,name='conv4')
        conv4=tf.nn.dropout(conv4,keep_prob=0.5,name='conv4_drop')
    with tf.variable_scope('max_pooling2') as scope:
        pool2=tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pooling2')

#第五层：卷积+RelU
    with tf.variable_scope('conv5') as scope:
        weights=tf.get_variable('weights',shape=[3,3,64,128],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[128],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(pool2,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv5=tf.nn.relu(conv,name='conv5')

#第六层：卷积+RelU+max_pooling
    with tf.variable_scope('conv6') as scope:
        weights=tf.get_variable('weights',shape=[3,3,128,128],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[128],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(conv5,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv6=tf.nn.relu(conv,name='conv6')
        conv6=tf.nn.dropout(conv6,keep_prob=0.5,name='conv6_drop')
    with tf.variable_scope('max_pooling3') as scope:
        pool3=tf.nn.max_pool(conv6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pooling3')

#第七层：卷积+RelU
    with tf.variable_scope('conv7') as scope:
        weights=tf.get_variable('weights',shape=[3,3,128,256],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[256],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(pool3,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv7=tf.nn.relu(conv,name='conv7')
    
#第八层：卷积+Relu
    with tf.variable_scope('conv8') as scope:
        weights=tf.get_variable('weights',shape=[3,3,256,256],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[256],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(conv7,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv8=tf.nn.relu(conv,name='conv8')
        conv8=tf.nn.dropout(conv8,keep_prob=0.5,name='conv8_drop')

#第九层：上采样（反卷积）+copy and crop
    with tf.variable_scope('deconv9') as scope:
        weights=tf.get_variable('weights',shape=[3,3,128,256],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        deconv9=tf.nn.conv2d_transpose(conv8,weights,tf.shape(conv6),strides=[1,2,2,1],padding='SAME')
        deconv9=tf.concat([deconv9,conv6],axis=3,name='deconv9')

#第十层：卷积
    with tf.variable_scope('conv10') as scope:
        weights=tf.get_variable('weights',shape=[3,3,256,128],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[128],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(deconv9,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv10=tf.nn.relu(conv,name='conv10')

#第十一层：卷积
    with tf.variable_scope('conv11') as scope:
        weights=tf.get_variable('weights',shape=[3,3,128,128],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[128],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(conv10,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv11=tf.nn.relu(conv,name='conv11')
        conv11=tf.nn.dropout(conv11,keep_prob=0.5,name='conv11_drop')



#第十二层：上采样（反卷积）+copy and crop
    with tf.variable_scope('deconv12') as scope:
        weights=tf.get_variable('weights',shape=[3,3,64,128],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        deconv12=tf.nn.conv2d_transpose(conv11,weights,tf.shape(conv4),strides=[1,2,2,1],padding='SAME')
        deconv12=tf.concat([deconv12,conv4],axis=3,name='deconv12')

#第十三层：卷积
    with tf.variable_scope('conv13') as scope:
        weights=tf.get_variable('weights',shape=[3,3,128,64],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[64],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(deconv12,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv13=tf.nn.relu(conv,name='conv13')

#第十四层：卷积
    with tf.variable_scope('conv14') as scope:
        weights=tf.get_variable('weights',shape=[3,3,64,64],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[64],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(conv13,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv14=tf.nn.relu(conv,name='conv14')
        conv14=tf.nn.dropout(conv14,keep_prob=0.5,name='conv14_drop')



#第十五层：上采样（反卷积）+copy and crop
    with tf.variable_scope('deconv15') as scope:
        weights=tf.get_variable('weights',shape=[3,3,32,64],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        deconv15=tf.nn.conv2d_transpose(conv14,weights,tf.shape(conv2),strides=[1,2,2,1],padding='SAME')
        deconv15=tf.concat([deconv15,conv2],axis=3,name='deconv15')

#第十六层：卷积
    with tf.variable_scope('conv16') as scope:
        weights=tf.get_variable('weights',shape=[3,3,64,32],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[32],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(deconv15,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv16=tf.nn.relu(conv,name='conv16')

#第十七层：卷积
    with tf.variable_scope('conv17') as scope:
        weights=tf.get_variable('weights',shape=[3,3,32,32],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[32],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(conv16,weights,strides=[1,1,1,1],padding='SAME')+biases
        conv17=tf.nn.relu(conv,name='conv17')
        conv17=tf.nn.dropout(conv17,keep_prob=0.5,name='conv17_drop')    
    
    
#第十八层：卷积
    with tf.variable_scope('conv18') as scope:
        weights=tf.get_variable('weights',shape=[3,3,32,1],dtype=tf.float32,
                                initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases=tf.get_variable('biases',shape=[1],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv18=tf.nn.conv2d(conv17,weights,strides=[1,1,1,1],padding='SAME')+biases
#        max_axis = tf.reduce_max(conv18, axis=3, keepdims=True)
#        exponential_map = tf.exp(conv18 - max_axis)
#        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
#    return exponential_map / normalize       
    return conv18



def losses(logits,labels):
    with tf.variable_scope('lose') as scope:
        cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy')
#        loss=-tf.reduce_mean(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)), name="cross_entropy")
        loss=tf.reduce_mean(cross_entropy,name='loss')
        tf.summary.scalar('lose/loss',loss)
    return loss


def training(loss):
    with tf.name_scope('optimizer'):
        optimizer=tf.train.AdamOptimizer(1e-4).minimize(loss)
    return optimizer
