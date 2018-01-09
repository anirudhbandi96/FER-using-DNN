from Dataset import image_data
from Dataset import data
import numpy as np
a = image_data()
#a.read_jaffe_data()
a.read_ck_data()
a.data_details()


#a.data_details()
#a.print_data()

import tensorflow as tf
sess = tf.InteractiveSession()
classes = a.classes
x = tf.placeholder(tf.float32, shape=[None, 68 , 128])
y_ = tf.placeholder(tf.float32, shape=[None, classes])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2,2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(x, [-1,68,128,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5,16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##W_conv3 = weight_variable([5, 5, 32,64])
##b_conv3 = bias_variable([64])
##
##h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
##h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([17 * 32 * 32, 2048])
b_fc1 = bias_variable([2048])

h_pool3_flat = tf.reshape(h_pool2, [-1, 17*32*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([2048 , classes])
b_fc2 = bias_variable([classes])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for iters in [270]:
  number_of_exps = 3
  acc_array =[]
  while(number_of_exps > 0):
    print(number_of_exps)
    
    sess.run(tf.global_variables_initializer())
    n = 30
    k = iters
    i = 0
    count = 0
    length = a.get_training_samples_length()
    while(count < k):
      #print("value of i:")
      #print(length)
      #print(length)
      b = length-i
      if b > n:
        batch = (a.training.features[i:i+n],a.training.labels[i:i+n])
        i = i+n
      else:
        batch = (np.concatenate((a.training.features[i:], a.training.features[:n-b]), axis = 0), np.concatenate(( a.training.labels[i:] , a.training.labels[:n-b]) , axis = 0))
        i = n - b
      count += 1
      if count%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(count, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    ##f = list(y_conv.eval(feed_dict={
    ##    x: a.testing.features , y_: a.testing.labels, keep_prob: 1.0}))
    ##f = [list(f[i]) for i in range(len(f))]
    ##e = []
    ##labels = list(a.testing.labels)
    ##labels = [list(x) for x in labels]
    ##for c in range(len(f[0])):
    ##  b = []
    ##  for d in range(len(f[0])):
    ##    b.append(0)
    ##  e.append(b)
    ###print(f)
    ##max = [i.index(max(i))  for i in f]
    ##for c in range(len(f)):
    ##  e[labels[c].index(1)][max[c]] += 1
    ##
    ##for n in range(len(e)):
    ##  e[n] = [ float(x) / sum(e[n]) for x in e[n]]
    ##
    ##print(e)
    ##acc = 0.0
    ##for q in range(len(e)):
    ##	acc += e[q][q]
    ##acc = acc/7

    ##print(acc)

    
   
    acc_array.append(accuracy.eval(feed_dict={x: a.testing.features , y_: a.testing.labels, keep_prob: 1.0}))
    number_of_exps -= 1
    print(acc_array)
  print("iter value is ", iters ,sum(acc_array)/len(acc_array))
    
