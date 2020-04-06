#coding=utf-8
import tensorflow as tf
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from collections import deque
from math import *
VECTOR_SIZE = 3


def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouse_ratio(mouse):
    # print(eye)
    A = distance.euclidean(mouse[1], mouse[11])
    B = distance.euclidean(mouse[4], mouse[8])
    C = distance.euclidean(mouse[0], mouse[6])
    mouse = (A + B) / (2.0 * C)
    return mouse


def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def img_round(image, degree=0):
    height, width = image.shape[:2]
    # degree = 90
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步
    image = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return image


# 神经网络参数
IMG_W = 96
IMG_H = 32
max_steps = 8000
batch_size = 1
CAPACITY = 1000 + 3 * batch_size
address = 'reload2/newmodel5.21'
eye_flag = 0


# 对眼部图像进行睁眼闭眼的分类 输入眼部图像 返回0(睁眼），1（闭眼）
def eye_evaluate(img1):
    # plt.imshow(img1)
    # plt.show()
    global eye_flag
    # print('image exists')
    with tf.Graph().as_default():
        image_hold1 = tf.placeholder(tf.float32, [IMG_H, IMG_W, 3])
        image_hold2 = tf.placeholder(tf.float32, [1, IMG_H, IMG_W, 3])
        # 转化图片格式
        # 图片标准化
        image_array = tf.cast(image_hold1, tf.float32)
        # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
        image_array = tf.image.adjust_brightness(image_hold1, +0.33)
        image_array = tf.image.per_image_standardization(image_array)
        image_array = tf.clip_by_value(image_array, 0.0, 1.0)
        image_batch = tf.reshape(image_array, [1, IMG_H, IMG_W, 3])

        weight1 = variable_with_weight_loss(shape=[5, 5, 3, 12], stddev=5e-2, wl=0.0)
        kernel1 = tf.nn.conv2d(image_hold2, weight1, [1, 1, 1, 1], padding='SAME')
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
        logit = tf.nn.softmax(logits)

        # 定义saver
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        logs_train_dir = address
        # print("从指定的路径中加载模型。。。。")
        # 将模型加载到sess 中
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            # print('模型加载成功, 训练的步数为 %s' % global_step)
        else:
            print('模型加载失败，，，文件没有找到')
        #opencv pil 图片尺寸都是W*H 而tensorflow是H*W
        print(img1.shape)

        img1 = cv2.resize(img1, (IMG_W, IMG_H))
        image_batch = sess.run(image_batch, feed_dict={image_hold1: img1})
        image_batch[0, :, :, 0] *= 0.3  #转化为更好识别的RGB值
        image_batch[0, :, :, 1] *= 0.59
        image_batch[0, :, :, 2] *= 0.11
        img_show = np.uint8(image_batch[0, :, :, :] * 255.0)
        cv2.imshow("eye2", img_show)
        cv2.waitKey(10)
        # plt.imshow(np.uint8(image_batch[0, :, :, :] * 255.0))
        # plt.show()
        prediction = sess.run(logit,  feed_dict={image_hold2: image_batch})
        # 获取输出结果中最大概率的索引
        max_index = np.argmax(prediction)
        # print(prediction)
        if max_index == 0:
            print('睁眼的概率 %.6f' % prediction[:, 0])
            eye_flag = 0
            return 0
        else:
            print('闭眼的概率 %.6f' % prediction[:, 1])
            eye_flag = 1
            return 1

pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

pts = deque(maxlen=100)
t = 0
i = 0.0
flag = 0
testflag = 0
location = 0
nodcount = 0
mousecount = 0
# 导入模型

#眉眼区域定位参数
LEFT_POINTS = 26
RIGHT_POINTS = 17
TOP_POINTS = 19
CENTRE_POINTS = 29

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

MOUSE_BEGIN = 48
MOUSE_END = 60
mframe_counter = 0
framecounter = 0
blinkcounter = 0
lblinkcounter = 0
state = 0
eye_state = 0
mouse_state = 0
nod_state = 0
eye_mean = 0
mouseEAR_sum_mean = 0
ear_vector = []

mouseEAR_sum = 0
ear_sum = 0

degree = 90
w1 = 0
w2 = 0
w3 = 0
r1 = 0
r2 = 0
imgfailure = 0
imgsuccess = 0
face_get_list = []
degree_change_flag = 0
degree_success_flag = 0



def clean():
    global blinkcounter, nodcount, state
    blinkcounter = 0
    nodcount = 0
    state = 0


def Image_evaluate(img):
    global pts, t, i, flag, location, nodcount, mousecount, mframe_counter, framecounter, blinkcounter
    global lblinkcounter, state, eye_state, mouse_state, nod_state, eye_mean, mouseEAR_sum_mean, ear_vector
    global mouseEAR_sum, ear_sum, w1, w2, w3, r1, r2, imgfailure, imgsuccess, degree,degree_change_flag, degree_success_flag
    if np.all(img == None):
        imgfailure += 1
        print('imgfailure:{}'.format(imgfailure))
        return 0, 0, 0
    else:
        imgsuccess += 1
        print('imgsucess:{}'.format(imgsuccess))
    face_get_list.append(i)

    if degree_change_flag == 1 and not(face_get_list[-1] == face_get_list[-2]):
        degree_success_flag = 1
    if degree_success_flag == 0 and len(face_get_list) > 1 and face_get_list[-1] == face_get_list[-2]:
        degree += 90
        degree_change_flag = 1
    img = img_round(img, degree)
    img = Contrast_and_Brightness(1.5, 4, img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detector(gray, 0)
    # print(i)
    print('-' * 20)
    for rect in rects:
        print("人脸数：{}".format(i))
        i = i + 1
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)
        # points = shape.parts()
        head = points[28]
        # 眉眼区域定位坐标
        lpoint = points[LEFT_POINTS]
        rpoint = points[RIGHT_POINTS]
        cpoint = points[CENTRE_POINTS]
        tpoint = points[TOP_POINTS]
        eye_img = img[tpoint[1]:cpoint[1], rpoint[0]:lpoint[0]]
        # b, g, r = cv2.split(eye_img)
        # eye_img2 = cv2.merge([r, g, b])
        cv2.imshow("eye", eye_img)
        cv2.waitKey(10)

        pts.appendleft(head)  # 获取头部中心点的位置加入到队列中
        t = len(pts) - 1
        mouse = points[MOUSE_BEGIN:MOUSE_END + 1]
        # 眼部大小
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouseEAR = mouse_ratio(mouse)
        ear = (leftEAR + rightEAR) / 2.0
        print(ear)
        if mouseEAR > 0.52:
            mframe_counter += 1
        else:
            if mframe_counter >= 5:
                mousecount += 1
            mframe_counter = 0
        if (i <= 200) or (eye_mean >= 0.3):
            if (mousecount/i) >= 1.0/3600:   #3分钟
                mouse_state = 1
            else:
                mouse_state = 0
            if (nodcount/i) >= 1.0/3600:
                nod_state = 1
            else:
                nod_state = 0

        if eye_evaluate(eye_img) == 1:
            framecounter += 1
        else:
            if framecounter >= 3:
                blinkcounter += 1
                if framecounter >= 6:
                    lblinkcounter += 1
                    if i>30:

                        w1 = framecounter / 30.0 + 0.25 + r1
                        w2 = (1 - w1) / 2 - r2 * r1
                        w3 = (1 - w1) / 2 - r1 * (1 - r2)
                        state = w1 * eye_state + w2\
                            * mouse_state + w3 * nod_state
                        print("%f * %d + %f * %d + %f * %d" % (w1, eye_state, w2, mouse_state, w3, nod_state))
                        if state > 1:
                            state = 1
                    if framecounter >= 14:
                        state = 1  #大于0.7s直接弹出
            framecounter = 0
        if i % 600 == 0:
            if lblinkcounter >= 5 and blinkcounter >= 11:
                eye_state = 1
            else:
                eye_state = 0

        if i<= 30:
            ear_sum += ear
            mouseEAR_sum += mouseEAR
            cv2.putText(img, "MouseSize: {:.2f}".format(mouseEAR), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 144, 30), 2)
            cv2.putText(img, "EyeSize: {:.2f}".format(ear), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 144, 30), 2)
            cv2.putText(img, "Headlocationw:{}".format(pts[0][1]), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 144, 30),2)
        else:
            eye_mean = ear_sum/30.0
            mouseEAR_sum_mean = mouseEAR_sum/30.0
            r1 = 2 * (eye_mean - 0.34)
            r2 = 2 * (mouseEAR_sum_mean - 0.35)
            if r1 > 0.1:
                r1 = 0.1
            elif r1 < -0.1:
                r1 = -0.1
            if r2 > 0.1:
                r2 = 0.1
            elif r2 < -0.1:
                r2 = -0.1
            cv2.putText(img, "EyeSizeAverage: {:.2f}".format(eye_mean), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 144, 30), 2)
            cv2.putText(img, "MouseSizeAverage:{:.2f}".format(mouseEAR_sum_mean), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 144, 30),
                        2)
            Freq = (blinkcounter / (i / 20.0)) * 60   #频率 单位：次/min
            cv2.putText(img, "Blinks: {0}".format(blinkcounter), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
            cv2.putText(img, "State: {:.2f}".format(state), (340, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
            cv2.putText(img, "Mouse:{0}".format(mousecount), (120, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
            cv2.putText(img, "Nod:{:.2f}".format(nodcount), (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
    if t > 10:
        if flag == 0:

            if pts[0][1] >= pts[2][1]:  # 新的低一些  y轴越低pts[0][1]的值越大
                pass
            else:
                relative = pts[0][1] - pts[t][1]
                # print("relative={0}".format(relative))
                if relative >= 10:
                    flag = 1
                    location = pts[t][1]
        elif flag == 1:
            if pts[0][1] <= pts[2][1]:
                pass
            else:
                if pts[0][1] <= location + 10:
                    nodcount += 1
                flag = 0
    cv2.imshow("Frame", img)
    cv2.waitKey(10)

    return blinkcounter, nodcount, int(state)


if __name__=="__main__":

    cap = cv2.VideoCapture(0)
    while(1):
        ret, img = cap.read()
        a, b, c = Image_evaluate(img)
        print(a, b, c)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



# if __name__=="__main__":
#     path = 'image/'
#     imagelist = os.listdir(path)
#     for imgname in imagelist:
#         if (imgname.endswith(".jpg")):
#             image = cv2.imread(path + imgname)
#
#
#             # cv2.imshow("picture", image)
#             Image_evaluate(image)
            # 每张图片的停留时间

            # 通过esc键终止程序

