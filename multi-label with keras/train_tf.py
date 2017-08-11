#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:50:44 2017

@author: jingang
"""

from getdata import load
import tensorflow as tf
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2



x_train, x_test, y_train, y_test = load()


x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')


x_train /= 255
x_test /= 255

x_train=np.transpose(x_train,(0,2,3,1))
x_test=np.transpose(x_test,(0,2,3,1))

# change the dimension of training and testing images


# define img size
img_size=100
img_shape=(img_size, img_size)
num_channels=3
num_classes=5





x=tf.placeholder(dtype=tf.float32, shape=[None,img_size, img_size,num_channels],name="x")

y_true=tf.placeholder(dtype=tf.float32, shape=[None,num_classes], name="y_true")


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape))

def new_bias(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))



def new_conv_layer(input,              #the previous layer
                   num_input_channels, #Num channels in prev. layer
                   filter_size,        #width and height of each filter
                   num_filters,        #NUmber of filters
                   use_pooling=True):  #Use 2x2 max-pooling
                   
                   
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_bias(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1],
                         padding='SAME')
    layer += biases
    
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    print("new_conv_layer")
    print(layer.get_shape())
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    
    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, [-1, num_features])
    print("flatten_layer")
    print(layer_flat.get_shape())
    print(num_features)
    return layer_flat, num_features

def new_fc_layer(input,      # the previous layer
                 num_inputs, # num inputs from previous layer
                 num_outputs,# num .outputs
                 use_relu =True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_bias(length=num_outputs)
    
    layer = tf.matmul(input, weights)+ biases
    
    if use_relu==True:
        layer = tf.nn.relu(layer)
    print("new_fc_layer")
    print(layer.get_shape())
    return layer

def drop_out(input, keep_prob):
   
    drop=tf.nn.dropout(input, keep_prob)
    print("dropout_layer")
    print(drop.get_shape())
    return drop

   
layer_conv1, weights_conv1 = new_conv_layer(input=x, num_input_channels=num_channels,
                                            filter_size=3, num_filters=32, use_pooling=False)

layer_conv2, weigths_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=32, 
                                            filter_size=3, num_filters=32, use_pooling=True)


drop_out1 = drop_out(layer_conv2, 0.75)

layer_conv3, weights_conv3 = new_conv_layer(input=drop_out1, num_input_channels=32,
                                           filter_size=3, num_filters=64, use_pooling=False)


layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3, num_input_channels=64,
                                           filter_size=3, num_filters=64, use_pooling=True)

drop_out2 = drop_out(layer_conv4, 0.75)  


layer_flat1, num_features1 = flatten_layer(drop_out2)
layer_fc1 = new_fc_layer(input=layer_flat1, num_inputs=num_features1, 
                         num_outputs=512, use_relu=True)


drop_out3 = drop_out(layer_fc1, 0.5)
layer_fc2 = new_fc_layer(input=drop_out3, num_inputs=512, 
                         num_outputs=5, use_relu=False)

y_pred=tf.nn.sigmoid(layer_fc2)
y_pred_cls=tf.round(y_pred)


cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost=tf.reduce_mean(cross_entropy)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
     

''' 
reward correct, punish incorrect
'''
correct_prediction=tf.equal(y_pred_cls, y_true)
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session=tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.global_variables_initializer())


train_batch_size = 64
total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    
    start_time = time.time()
    
    
    
    for i in range(total_iterations, total_iterations+num_iterations):
        
        index=np.random.randint(low=0, high=1599, size=train_batch_size)
        x_batch=x_train[index,:,:,:]
        y_true_batch=y_train[index,:]
        
        feed_dic_train = {x:x_batch, y_true:y_true_batch}
        



        session.run(optimizer, feed_dict=feed_dic_train)
        
        if i%100 ==0:
            acc = session.run(accuracy, feed_dict=feed_dic_train)
            msg = "Optimization Iteration:{0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i+1, acc))
    total_iterations += num_iterations
        
    end_time=time.time()
    time_dif = end_time-start_time

    print ("Time usage: "+ str(timedelta(seconds=int(round(time_dif)))))    


test_batch_size = 40


def print_test_accuracy( show_example_errors=False):
    num_test = len(y_test)
    
    cls_pred = np.zeros(shape=[num_test,num_classes], dtype=np.int)
    
    i = 0
    
    while i < num_test:
        
        j = min(i+test_batch_size, num_test)
        
        images = x_test[i:j,:,:,:]
        
        labels = y_test[i:j,:]
        
        feed_dict = {x:images, y_true:labels}
        
        cls_pred[i:j,:] = session.run(y_pred_cls, feed_dict=feed_dict)
        
        i=j
       
    correct = (cls_pred == y_test)
    h,w = correct.shape
    correct_vec = np.ones(h)
    for i in range(h):
        for j in range(w):
            if correct[i,j]==0:
                correct_vec[i]=0
    correct_sum = correct.sum()
    acc=float(correct_sum)/num_classes/num_test   
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct_vec)
    
    
    

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 4
    
    
   
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    for i, ax in enumerate(axes.flat):
      
        ax.imshow(images[i],cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i,:])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i,:], cls_pred[i,:])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    fig.suptitle("desert,mountain,sea,sunset,tree")
    plt.show()


    
def plot_example_errors(cls_pred, correct):
    incorrect = (correct==False)
    images = x_train[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = y_train[incorrect]  
    
    plot_images(images=images[0:4],
                cls_true=cls_true[0:4],
                cls_pred=cls_pred[0:4])

    
optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=True)    

saver = tf.train.Saver()
save_path = saver.save(session, "model_tf.ckpt")
print("Model saved in file: %s" % save_path)


'''
image_path="im1.jpg"
test_image=Image.open(image_path)
test_image= test_image.resize((100,100), PIL.Image.ANTIALIAS)
plt.imshow(test_image)
test_image.save('test.jpg')



test_image=cv2.imread('test.jpg')
test_image= test_image.astype('float32')/255
plt.imshow(test_image)
test_image=[test_image]
label=np.array([[0,0,0,0,0]])
feed={x: test_image, y_true:label}
print(session.run(y_pred,feed_dict=feed))
'''