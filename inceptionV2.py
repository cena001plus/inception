​
import tensorflow as tf
import numpy as np


#定义变量函数初始化函数
def define_variable(shape,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name)


# 最大下采样操作
def max_pool(name, l_input, k1, k2):
    return tf.nn.max_pool(l_input, ksize=[1, k1, k1, 1], strides=[1, k2, k2, 1], padding='SAME', name=name)


# network structure
def inception_model(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    p1f11 = filters_1x1
    p2f11 = filters_3x3_reduce
    p2f33 = filters_3x3
    p3f11 = filters_5x5_reduce
    p3f55 = filters_5x5
    p4f11 = filters_pool_proj

    path1 = tf.layers.conv2d(input, p1f11, 1, padding='same', activation=tf.nn.relu)

    path2 = tf.layers.conv2d(input, p2f11, [1, 1], activation=tf.nn.relu)
    path2 = tf.layers.conv2d(path2, p2f33, [1, 3], padding='same', activation=tf.nn.relu)
    path2 = tf.layers.conv2d(path2, p2f33, [3, 1], padding='same', activation=tf.nn.relu)

    path3 = tf.layers.conv2d(input, p3f11, 1, activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f55, [1, 3], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f55, [3, 1], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f55, [1, 3], padding='same', activation=tf.nn.relu)
    path3 = tf.layers.conv2d(path3, p3f55, [3, 1], padding='same', activation=tf.nn.relu)

    path4 = tf.layers.max_pooling2d(input, pool_size=3, strides=1, padding='same')
    path4 = tf.layers.conv2d(path4, p4f11, 1, activation=tf.nn.relu)
    out = tf.concat((path1, path2, path3, path4), axis=-1)
    return out


# 网络结构定义
# 输入参数：images，image batch、4D tensor、tf.float32、[batch_size, width, height, channels]
# 返回参数：logits, float、 [batch_size, n_classes]
def inference(images,batch_size,  n_classes):
    W_conv1 = define_variable([3, 3, 3, 192], name="W")
    b_conv1 = define_variable([192], name="B")
    conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + b_conv1)

    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')

    inception_3a = inception_model(input=pool1, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
                                   filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)
    inception_3b = inception_model(input=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                                   filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)

    maxpool3_3x3_s2 = tf.nn.max_pool(inception_3b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    inception_4a = inception_model(input=maxpool3_3x3_s2, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
                                   filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)
    inception_4b = inception_model(input=inception_4a, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224,
                                   filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4c = inception_model(input=inception_4b, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256,
                                   filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4d = inception_model(input=inception_4c, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288,
                                   filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64)
    inception_4e = inception_model(input=inception_4d, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                                   filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    maxpool4_3x3_s2 = tf.nn.max_pool(inception_4e, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    inception_5a = inception_model(input=maxpool4_3x3_s2, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320,
                                   filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    inception_5b = inception_model(input=inception_5a, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384,
                                   filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128)

    net = tf.layers.average_pooling2d(inception_5b, 7, 1, name="avgpool")  # -> [batch, 1, 1, 768]

    reshape = tf.reshape(net, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights1 = tf.Variable(tf.truncated_normal(shape=[dim, 1024], stddev=0.005, dtype=tf.float32),name='weights', dtype=tf.float32)
    biases1 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[1024]),name='biases', dtype=tf.float32)
    local6 = tf.nn.relu(tf.matmul(reshape, weights1) + biases1)

    weights = tf.Variable(tf.truncated_normal(shape=[1024, n_classes], stddev=0.005, dtype=tf.float32),name='softmax_linear', dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),name='biases', dtype=tf.float32)
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