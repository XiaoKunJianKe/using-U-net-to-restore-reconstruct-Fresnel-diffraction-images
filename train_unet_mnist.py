# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:43:43 2019

@author: XiaoKunJianKe
"""
import tensorflow as tf
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import mode
train_image_dir1='E:/Mnist_32_32/train_image_'
train_label_dir1='E:/Mnist_32_32/train_label_'
logs_train_dir='E:/Mnist_32_32/log/'
batch_size=2
MAX_STEP=25000
image_batch, label_batch= mode.get_pair_batch(train_image_dir1,train_label_dir1,batch_size)
train_logits=mode.train(image_batch)
train_loss=mode.losses(train_logits,label_batch)
train_op=mode.training(train_loss)

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess = sess, coord = coord)     
     summary_op=tf.summary.merge_all()
     train_writer=tf.summary.FileWriter(logs_train_dir, sess.graph)                                           
     saver=tf.train.Saver()
     try:         
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
               break
            _, tra_loss = sess.run([train_op, train_loss])        
            if step % 100 == 0:
               print('Step %d, train loss = %.2f' %(step, tra_loss))
               summary_str = sess.run(summary_op)
               train_writer.add_summary(summary_str, step) 
            if step % 4000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)
     except tf.errors.OutOfRangeError:        
         print('Done training -- epoch limit reached')
     finally:
          coord.request_stop()   
     coord.join(threads)
     sess.close()
    




























        
        
       





































       
      
        
                                                                                

        
        
        
         







    
    
    
