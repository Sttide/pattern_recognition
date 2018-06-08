# Time    : 18-5-16 上午10:40
# Email   : sttide@outlook.com

import tensorflow as tf
import work3_dataset
from numpy import *
import os


#保存模型
model_dir = "model/faceclass"
model_name = "faceclass"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

#超参数
batch_size = 128
epoch_num = int(720/batch_size)
epoch = 0
iters = 0

def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def forward(image,regularizer,keep_probs):
    W_conv1 = get_weight([5, 5, 1, 64],regularizer)
    b_conv1 = get_bias([64])

    x_image = tf.reshape(image, [-1,32,32,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print(h_pool1)

    ##第二层卷积
    W_conv2 = get_weight([5, 5, 64, 128],regularizer)
    b_conv2 = get_bias([128])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print(h_pool2)


    pool_shape = h_pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    print(nodes)
    reshaped = tf.reshape(h_pool2,[-1,nodes])

    fc1_w = get_weight([nodes,256],regularizer)
    fc1_b = get_bias([256])
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    fc1_d = tf.nn.dropout(fc1, keep_prob=keep_probs)

    fc2_w = get_weight([256,38], regularizer)
    fc2_b = get_bias([38])
    y = tf.nn.relu(tf.matmul(fc1_d,fc2_w)+fc2_b, name='op')

    return y

def backward(train_images, train_labels):
    x = tf.placeholder(tf.float32, [None, 1024], name='x')
    y_ = tf.placeholder(tf.float32, [None, 38], name='y_')
    keep_probs = tf.placeholder(tf.float32, name='keep_probs')
    global_step = tf.Variable(0, trainable=False)
    regularizer = 0.0001
    y = forward(x, regularizer,keep_probs)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    saver = tf.train.Saver()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accu')

    train_step = tf.train.AdamOptimizer(0.005).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(iters):
            _, loss_value, step, train_accuracy = sess.run([train_step, loss, global_step, accuracy],
                                                           feed_dict={x: train_images, y_: train_labels,keep_probs:0.5})
            if step % 1 == 0:
                if step == 0:
                    continue
                print("step %d, loss %g, training accuracy %.2f%%" % (step, loss_value, train_accuracy * 100))

        print("Training Success!")
        saver.save(sess, os.path.join(model_dir, model_name), global_step=0)
        print("Save success！")
        sess.close()

def decetec(test_images, test_labels ):

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/faceclass/faceclass-0.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('./model/faceclass/'))  # 只需要指定目录就可以恢复所有变量信息

    # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y_:0')
    input_kb = sess.graph.get_tensor_by_name('keep_probs:0')
    # 获取需要进行计算的operator
    test_accuracy = sess.graph.get_tensor_by_name('accu:0')

    #opt input_x input_keep一定要在mnist"name"属性定义好
    ret = sess.run(test_accuracy,feed_dict={input_x: test_images,input_y:test_labels, input_kb:0.5})
    print("Test accuracy: " ,ret*100, "%")
    return ret


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = work3_dataset.dataset()
    backward(train_images,train_labels)
    decetec(test_images,test_labels)
