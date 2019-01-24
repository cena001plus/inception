​
import tensorflow as tf
import numpy as np


# 定义变量函数初始化函数
def define_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name)

# 最大下采样操作
def max_pool(name, l_input, k1, k2):
    return tf.nn.max_pool(l_input, ksize=[1, k1, k1, 1], strides=[1, k2, k2, 1], padding='SAME', name=name)

# network structure: inception_A
def inception_A(input):
    p1f11 = 96
    p2f11 = 64
    p2f22 = 96
    p3f11 = 64
    p3f22 = 96
    p3f33 = 96
    p4f11 = 96
    path1 = tf.layers.conv2d(input, p1f11, 1, padding='same', activation=tf.nn.relu)
    path2 = tf.layers.conv2d(input, p2f11, 1, padding='same', activation=tf.nn.relu)
    path2 = tf.layers.conv2d(path2, p2f22, 3, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(input, p3f11, 1, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f22, 3, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f33, 3, padding='same', activation=tf.nn.relu)
    path4 = tf.layers.average_pooling2d(input, pool_size=3, strides=1, padding='same')
    path4 = tf.layers.conv2d(path4, p4f11, 1, padding='same', activation=tf.nn.relu)
    out = tf.concat((path1, path2, path3, path4), axis=-1)
    return out

# network structure: Reduction_A
def reduction_A(input):
    channel = 384
    path1 = tf.layers.max_pooling2d(input, pool_size=3, strides=2, padding='same')
    path2 = tf.layers.conv2d(input, channel, 3, strides=2, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(input, channel, 1, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, channel, 3, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, channel, 3, strides=2, padding='same', activation=tf.nn.relu)
    out = tf.concat((path1, path2, path3), axis=-1)
    return out

# network structure: inception_B
def inception_B(input):
    p1f11 = 384
    p2f11 = 192
    p2f22 = 224
    p2f33 = 256
    p3f11 = 192
    p3f22 = 192
    p3f33 = 224
    p3f44 = 224
    p3f55 = 256
    p4f11 = 128
    path1 = tf.layers.conv2d(input, p1f11, 1, padding='same', activation=tf.nn.relu)
    path2 = tf.layers.conv2d(input, p2f11, 1, padding='same', activation=tf.nn.relu)
    path2 = tf.layers.conv2d(path2, p2f22, [1, 7], padding='same',activation=tf.nn.relu)
    path2 = tf.layers.conv2d(path2, p2f33, [7, 1], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(input, p3f11, 1, padding='same',activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f22, [1, 7], padding='same',activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f33, [7, 1], padding='same',activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f44, [1, 7], padding='same',activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f55, [7, 1], padding='same',activation=tf.nn.relu)
    path4 = tf.layers.average_pooling2d(input, pool_size=3, strides=1, padding='same')
    path4 = tf.layers.conv2d(path4, p4f11, 1, padding='same',activation=tf.nn.relu)
    out = tf.concat((path1, path2, path3, path4), axis=-1)
    return out

# network structure: Reduction_B
def reduction_B( input):
    p2f11 = 192
    p2f22 = 192
    p3f11 = 256
    p3f22 = 256
    p3f33 = 320
    p3f44 = 320
    path1 = tf.layers.max_pooling2d(input, pool_size=3, strides=2, padding='same')
    path2 = tf.layers.conv2d(input, p2f11, 1,  padding='same', activation=tf.nn.relu)
    path2 = tf.layers.conv2d(path2, p2f22, 3, strides=2, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(input, p3f11, 1, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f22, [1, 7], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f33, [7, 1], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f44, 3, strides=2, padding='same', activation=tf.nn.relu)
    out = tf.concat((path1, path2, path3), axis=-1)
    return out

# network structure: inception_C
def inception_C(input):
    p1f11 = 256
    p2f11 = 384
    p2f11_1 = 256
    p2f11_2 = 256
    p3f11 = 384
    p3f22 = 448
    p3f33 = 512
    p3f33_1 = 256
    p3f33_2 = 256
    p4f11 = 256
    path1 = tf.layers.conv2d(input, p1f11, 1, padding='same', activation=tf.nn.relu)
    path2 = tf.layers.conv2d(input, p2f11, 1, padding='same',activation=tf.nn.relu)
    path2_1 = tf.layers.conv2d(path2, p2f11_1, [1, 3], padding='same', activation=tf.nn.relu)
    path2_2 = tf.layers.conv2d(path2, p2f11_2, [3, 1], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(input, p3f11, 1, padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f22, [1, 3], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f33, [3, 1], padding='same', activation=tf.nn.relu)
    path3_1 = tf.layers.conv2d(path3, p3f33_1, [3, 1], padding='same', activation=tf.nn.relu)
    path3_2 = tf.layers.conv2d(path3, p3f33_2, [1, 3], padding='same', activation=tf.nn.relu)
    path4 = tf.layers.average_pooling2d(input, pool_size=3, strides=1, padding='same')
    path4 = tf.layers.conv2d(path4, p4f11, 1, padding='same', activation=tf.nn.relu)
    out = tf.concat((path1, path2_1, path2_2, path3_1, path3_2, path4), axis=-1)
    return out


# 网络结构定义
# 输入参数：images，image batch、4D tensor、tf.float32、[batch_size, width, height, channels]
# 返回参数：logits, float、 [batch_size, n_classes]
def inference(images, batch_size,  n_classes):
    w_conv1 = define_variable([3, 3, 3, 192], name="W")
    b_conv1 = define_variable([192], name="B")
    conv1 = tf.nn.conv2d(images, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + b_conv1)

    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

    # inception_A
    # 35*35 grid
    inception_a1 = inception_A(pool1)
    inception_a2 = inception_A(inception_a1)
    inception_a3 = inception_A(inception_a2)
    inception_a4 = inception_A(inception_a3)

    # reduction_A
    # from 35*35 to 17*17
    reduction_a = reduction_A(inception_a4)

    # inception_B
    # 17*17 grid
    inception_b1 = inception_B(reduction_a)
    inception_b2 = inception_B(inception_b1)
    inception_b3 = inception_B(inception_b2)
    inception_b4 = inception_B(inception_b3)
    inception_b5 = inception_B(inception_b4)
    inception_b6 = inception_B(inception_b5)
    inception_b7 = inception_B(inception_b6)

    # reduction_B
    # from 17*17 to 8*8
    reduction_b = reduction_B(inception_b7)

    # inception_C
    # 8*8 grid
    inception_c1 = inception_C(reduction_b)
    inception_c2 = inception_C(inception_c1)
    inception_c3 = inception_C(inception_c2)

    net = tf.layers.average_pooling2d(inception_c3, 7, 1, name="avgpool")  # -> [batch, 1, 1, 768]

    # dropout层
    with tf.variable_scope('dropout') as scope:
        drop_out = tf.nn.dropout(net, 0.8)

    reshape = tf.reshape(drop_out, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights1 = tf.Variable(tf.truncated_normal(shape=[dim, 1024], stddev=0.005, dtype=tf.float32),name='weights', dtype=tf.float32)
    biases1 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[1024]),name='biases', dtype=tf.float32)
    local6 = tf.nn.relu(tf.matmul(reshape, weights1) + biases1)

    weights = tf.Variable(tf.truncated_normal(shape=[1024, n_classes], stddev=0.005, dtype=tf.float32), name='softmax_linear', dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]), name='biases', dtype=tf.float32)
    logits = tf.add(tf.matmul(local6, weights), biases, name='softmax_linear')

    return logits

# -----------------------------------------------------------------------------
# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# --------------------------------------------------------------------------
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# -----------------------------------------------------------------------
# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

​