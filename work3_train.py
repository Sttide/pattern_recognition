# -*- coding: utf-8 -*-
# Created Time    : 18-5-20 下午10:25
# Connect me with : sttide@outlook.com

import tensorflow as tf
from numpy import *
import work3_1


def restore_model_ckpt(test_image):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/faceclass/faceclass-0.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('./model/faceclass/'))  # 只需要指定目录就可以恢复所有变量信息


    # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('x:0')
    # 获取需要进行计算的operator
    opt = sess.graph.get_tensor_by_name('op:0')

    #opt input_x input_keep一定要在mnist"name"属性定义好
    ret = sess.run(opt,feed_dict={input_x: test_image})
    #print(ret*10000)
    res = argmax(ret,1)
    return res[0]

if __name__ == "__main__":
    test_images, test_labels = work3_1.test_images,work3_1.test_lables
    length = test_images
    test_labs = array(test_labels,dtype=int)
    print(test_labs)
    num = 0
    count = 0
    for i in test_images:
        test = reshape(i,[-1,1024])
        res_i = restore_model_ckpt(test)
        real_l = test_labs[num]
        num = num + 1
        print(real_l)
        real_r = real_l.argmax(0)
        print(real_r)
        if res_i == real_r:
            count = count + 1

    print("%.2f",(num/length)*100)

'''
ema = tf.train.ExponentialMovingAverage(0.99, global_step)
ema_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name='train')

    for i in range(iters):
        image, lables = shuffle(train_images, train_lables)
        for j in range(epoch_num):
            batch_img = image[(j*batch_size):((j+1)*batch_size)]
            bat_lab = lables[(j*batch_size):((j+1)*batch_size)]
            _, loss_value, step, train_accuracy = sess.run([train_step, loss, global_step, accuracy],
                                                           feed_dict={x: batch_img, y_: bat_lab})
            if step % 100 == 0:
                if step == 0:
                    continue
                print("step %d, loss %g, training accuracy %.2f%%" % (step, loss_value, train_accuracy*100))
'''
