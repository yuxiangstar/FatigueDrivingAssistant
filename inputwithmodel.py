# %%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import tensorflow as tf
import numpy as np
import time
import cv2
from sklearn import preprocessing
import matplotlib.pyplot as plt
# BATCH_SIZE = 32

IMG_W = 96 + 3
IMG_H = 32 + 3

IMG_WL = 96
IMG_HL = 32
max_steps = 0
batch_size = 64
CAPACITY = 1000 + 3 * batch_size
train_dir = 'photo/newsummary1.0/'
test_dir = 'photo/newtest2.0/'
def get_files(file_dir):
    open_eyes = []
    label_open_eyes = []
    close_eyes = []
    label_close_eyes = []
    imgs = os.listdir(file_dir)
    imgNum = len(imgs)
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == '1':
            close_eyes.append(file_dir  + file )
            label_close_eyes.append(1)
        else:
            if name[0] == '0':
                open_eyes.append(file_dir  + file)
                label_open_eyes.append(0)
        image_list = np.hstack((close_eyes, open_eyes))
        label_list = np.hstack((label_close_eyes, label_open_eyes))
    # print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))
    # 多个种类分别的时候需要把多个种类放在一起，打乱顺序,这里不需要

    # 把标签和图片都放倒一个 temp 中 然后打乱顺序，然后取出来
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    # 打乱顺序
    np.random.shuffle(temp)

    # 取出第一个元素作为 image 第二个元素作为 label
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    print(label_list)
    label_list = [int(i) for i in label_list]

    return image_list, label_list

def get_batch(image, label, image_H, image_W, batch_size, capacity):
    # 转换数据为 ts 能识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [image_H, image_W], method=0)
    #随机翻转数据增强
    image = tf.image.random_flip_left_right(image)

    image = tf.random_crop(image, [image_H-3, image_W-3, 3])
    image = tf.image.adjust_brightness(image, +0.33)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.per_image_standardization(image)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)

    # 重新定义下 label_batch 的形状
    label_batch = tf.reshape(label_batch, [batch_size])
    # 转化图片

    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch



def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def loss(logits, labels):
    #      """Add L2Loss to all the trainable variables.
    #      Add summary for "Loss" and "Loss/avg".
    #      Args:
    #        logits: Logits from inference().
    #        labels: Labels from distorted_inputs or inputs(). 1-D tensor
    #                of shape [batch_size]
    #      Returns:
    #        Loss tensor of type float.
    #      """
    #      # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


###


image_list, label_list = get_files(train_dir)
images_train, labels_train = get_batch(image_list, label_list, IMG_H, IMG_W, batch_size, CAPACITY)

image_list_test, label_list_test = get_files(test_dir)
images_test, labels_test = get_batch(image_list_test, label_list_test, IMG_H, IMG_W, batch_size, CAPACITY)

# images_train, labels_train = cifar10.distorted_inputs()
# images_test, labels_test = cifar10.inputs(eval_data=True)

image_holder = tf.placeholder(tf.float32, [batch_size, IMG_HL, IMG_WL, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# logits = inference(image_holder)

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 12], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[12]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

weight2 = variable_with_weight_loss(shape=[5, 5, 12, 32], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[32]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 240], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[240]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[240, 120], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[120]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[120, 2], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[2]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

loss = loss(logits, label_holder)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 0.72

top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
tf.train.start_queue_runners()
min_max_scaler = preprocessing.MinMaxScaler()
###

# 重新加载
logs_train_dir = 'reload2/newmodel5.21'
ckpt = tf.train.get_checkpoint_state(logs_train_dir)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    # print('模型加载成功, 训练的步数为 %s' % global_step)
else:
    print('模型加载失败，，，文件没有找到')

t1 = time.time()
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    # for index in range(batch_size):
    #     image_batch[index] = np.power(image_batch[index] / float(np.max(image_batch[index])), 1.5)
    #     image_gray_blur2 = cv2.GaussianBlur(image_batch[index], (3, 3), 0.3)
    #     image_gray_blur3 = cv2.GaussianBlur(image_gray_blur2, (3, 3), 0.4)
    #     image_batch[index] = image_gray_blur3 - image_gray_blur2
    #     for channel in range(3):
    #         image_batch[index, :, :, channel] = min_max_scaler.fit_transform(image_batch[index, :, :, channel])
    for a in range(batch_size):
        # for b in range(image_W):
        #     for c in range(image_W):
        image_batch[a, :, :, 0] *= 0.3
        image_batch[a, :, :, 1] *= 0.59
        image_batch[a, :, :, 2] *= 0.11
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch,
                                                          label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
        if step % 2000 == 0:
            saver.save(sess, 'reload2/newmodel5.22/model.ckpt', global_step=step)
###
t2 = time.time()
print("训练时长" + str((t2 - t1) / 60) + "分钟")

num_examples = 650
import math

num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    for a in range(batch_size):
        # for b in range(image_W):
        #     for c in range(image_W):
        image_batch[a, :, :, 0] *= 0.3
        image_batch[a, :, :, 1] *= 0.59
        image_batch[a, :, :, 2] *= 0.11
    # plt.imshow(np.uint8(image_batch[0, :, :, :] * 255.0))
    # plt.show()
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
