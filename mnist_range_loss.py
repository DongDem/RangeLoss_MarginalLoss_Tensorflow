LAMBDA = 0.01
# Parameters
NUM_CLASSES = 10

# Import modules
import os
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data
slim = tf.contrib.slim


# Construct Network
with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')

global_step = tf.Variable(0, trainable=False, name='global_step')

def pairwise_distance(array):
    # Take the size of the array
    size = tf.shape(array)[0]
    # Make 2D grid of indices
    r = tf.range(size)
    ii, jj = tf.meshgrid(r, r, indexing='ij')
    # Take pairs of indices where the first is less or equal than the second
    m = ii < jj
    idx = tf.stack([tf.boolean_mask(ii, m), tf.boolean_mask(jj, m)], axis=1)
    # Gather result
    result = tf.gather(array, idx)

    i = tf.constant(0)
    l = tf.Variable([])
    def condition(i, l):
        return tf.less(i, tf.shape(result)[0])
    def body(i, l):
        # do something here which you want to do in your loop
        # increment i
        temp=tf.nn.l2_loss(result[i][0] - result[i][1])
        l = tf.concat([l, [temp]], 0)
        return tf.add(i, 1),l

    # do the loop:
    _, list_val= tf.while_loop(condition, body, [i, l],shape_invariants=[i.get_shape(),
                                                   tf.TensorShape([None])])
    return list_val


def _compute_top_k(features):
    main_len_features = tf.constant(3)
    def f1():
        dist_array = pairwise_distance(features)
        m = tf.nn.top_k(dist_array, k=2, sorted=True)
        return m[0][0], m[0][1]
    def f2():
        return 0.,0.
    r = tf.cond(tf.less(tf.shape(features)[0], main_len_features), f2, f1)
    return r


def _compute_min_dist(center_features):
    min_number_centers = tf.constant(2)
    def f1():
        dist_array2 = pairwise_distance(center_features)
        m = tf.nn.top_k(dist_array2, k=tf.shape(dist_array2)[0], sorted=True)
        lowest_value = tf.gather(m[0], tf.shape(dist_array2)[0] - 1)
        return lowest_value
    def f2():
        return 0.
    r = tf.cond(tf.less(tf.shape(center_features)[0],min_number_centers),f2,f1)
    return r

def _calculate_centers(features, labels):
    unique_labels, idx = tf.unique(labels)
    j = tf.constant(0)
    output_list_1 = tf.Variable(tf.zeros([1,2]),dtype=tf.float32)
    def condition(j,output_list_1):
        return tf.less(j, tf.shape(unique_labels)[0])
    def body(j,output_list_1):
        label = unique_labels[j]
        index_array = tf.where(tf.equal(labels, label))
        index_array = tf.reshape(index_array,[-1])
        same_class_features = tf.gather(features,index_array)
        center_features_elements= tf.reduce_mean(same_class_features, axis=0)
        output_list_1 = tf.concat([output_list_1, [center_features_elements]],0)
        return tf.add(j, 1), output_list_1

    idx1,center_features = tf.while_loop(condition, body, [j, output_list_1],shape_invariants=[j.get_shape(),tf.TensorShape([None,None])])
    m = tf.range(1, tf.shape(center_features)[0], 1)
    center_feature = tf.gather(center_features,m)
    return center_feature

# Range Loss
def get_range_loss(features, labels):
    margin = 1.
    alpha = tf.constant(0.5)
    beta = tf.constant(0.5)
    epsilon=tf.constant(10e-5)

    # intra_class loss
    labels = tf.reshape(labels, [-1])
    unique_labels, idx = tf.unique(labels)
    output_list = tf.Variable([])
    i=tf.constant(0)
    def condition(i,output_list ):
        return tf.less(i,tf.shape(unique_labels)[0] )

    def body(i,output_list):
        label = unique_labels[i]
        index_array = tf.where(tf.equal(labels, label))
        index_array = tf.reshape(index_array,[-1])
        # get feature same class
        same_class_features = tf.gather(features,index_array)
        # caculate 2 largerst distance
        top_1, top_2 = _compute_top_k(same_class_features)
        same_class_distances = tf.add(tf.divide(1.,top_1+epsilon), tf.divide(1.,top_2+epsilon) )
        same_class_distances1 = tf.divide(2., same_class_distances)

        output_list = tf.concat([output_list, [same_class_distances1]], 0)
        return tf.add(i, 1),output_list

    idx, intra_distance = tf.while_loop(condition, body, [i, output_list],shape_invariants=[i.get_shape(),tf.TensorShape([None])])
    intra_class_loss = tf.reduce_sum(intra_distance)

    # inter_class loss
    # find center features
    center_features = _calculate_centers(features, labels)
    # find min distance among center features
    min_inter_class_center_distance = _compute_min_dist(center_features)
    inter_class_loss = tf.maximum((margin - min_inter_class_center_distance), 0.)
    # combine to get range loss
    range_loss = tf.add(tf.multiply(alpha,intra_class_loss), tf.multiply(beta,inter_class_loss))

    return range_loss

# CNN Architectures

def inference1(input_images):  # with slim.arg_scope([slim.conv2d],padding='SAME',weights_regularizer=slim.l2_regularizer(0.001)):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images, num_outputs=64, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')

            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')

            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')

            x = slim.flatten(x, scope='flatten')

            x = slim.fully_connected(x, num_outputs=1024, activation_fn=None, scope='fc1')
            x = tflearn.prelu(x)

            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc2')
            x = tflearn.prelu(feature)

            x = slim.dropout(x, scope='dropout2')

            x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc3')

    return x, feature
def inference(input_images):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images, num_outputs=32, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')

            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')

            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')

            x = slim.flatten(x, scope='flatten')

            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')

            x = tflearn.prelu(feature)

            x = slim.fully_connected(x, num_outputs=10, activation_fn=None, scope='fc2')

    return x, feature


def build_network(input_images, labels, ratio):
    logits, features = inference(input_images)

    with tf.name_scope('loss'):

        with tf.name_scope('range_loss'):
            range_loss = get_range_loss(features, labels)

        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            print(softmax_loss)
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * range_loss

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('RangeLoss', range_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)

    return logits, features, total_loss, accuracy, softmax_loss , range_loss

logits, features, total_loss, accuracy, softmax_loss ,range_loss= build_network(input_images, labels, ratio=LAMBDA)

# Prepare Data
mnist = input_data.read_data_sets('./tmp/mnist', reshape=False)

starter_learning_rate = 0.001
LEARNING_RATE_DECAY_FACTOR = 0.25
decay_steps = 5000
decayed_learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   LEARNING_RATE_DECAY_FACTOR,
                                                   staircase=True)
# Optimizer
optimizer = tf.train.RMSPropOptimizer(decayed_learning_rate)

train_op = optimizer.minimize(total_loss, global_step=global_step)
predict =tf.argmax(logits,1)
# Seesion and Summary

logs_path = "./logdir"
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)
#sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Initialize the variables
writer_test = tf.summary.FileWriter("./logdir/plot_test")
writer_train = tf.summary.FileWriter("./logdir/plot_train")
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("cost", total_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, sess.graph)

# Train
mean_data = np.mean(mnist.train.images, axis=0)
step = sess.run(global_step)
while step <= 7000:
    batch_images, batch_labels = mnist.train.next_batch(64)
    print(batch_labels)
    _, summary_str, train_acc,printed_loss,printed_range_loss,printed_softmax_loss = sess.run([train_op, summary_op, accuracy, total_loss,range_loss,softmax_loss],feed_dict={input_images: batch_images - mean_data,labels: batch_labels})
    step += 1
    print("Total_Loss: {:.6f}".format(printed_loss))
    print("Range_Loss: {:.6f}".format(printed_range_loss))
    print("Softmax_Loss: {:.6f}".format(printed_softmax_loss))
    writer_train.add_summary(summary_str, global_step=step)

    if step % 50 == 0:
        vali_image = mnist.validation.images - mean_data
        vali_acc = sess.run( accuracy,feed_dict={input_images: vali_image,labels: mnist.validation.labels})
        writer_test.add_summary(summary_str, global_step=step)
        print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".format(step, train_acc, vali_acc)))
# Visualize train_data

feat = sess.run(features, feed_dict={input_images:mnist.train.images[:10000]-mean_data})

#matplotlib inline
import matplotlib.pyplot as plt

labels = mnist.train.labels[:10000]

f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()

# Visualize test_data
feat = sess.run(features, feed_dict={input_images:mnist.test.images[:10000]-mean_data})

#matplotlib inline
import matplotlib.pyplot as plt

labels = mnist.test.labels[:10000]

f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()
saver = tf.train.Saver()
save_path = saver.save(sess, "./model/mnist_with_range_loss.ckpt")