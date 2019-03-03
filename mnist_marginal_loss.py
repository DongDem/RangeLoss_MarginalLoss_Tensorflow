LAMBDA = 0.01
# Parameters
NUM_CLASSES = 10

# Import modules
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

# Marginal Loss
def get_marginal_loss(features, labels):
    '''
    Arguments:
        features: Tensor, [batch_size, feature_length]
        labels: Tensor, [batch_size]
    Return:
        loss: marginal loss
    '''

    theta = tf.constant(1.2, dtype=tf.float32)
    epsilon = tf.constant(0.3, dtype=tf.float32)

    features= tf.divide(features, tf.norm(features, ord='euclidean'))

    size = tf.shape(features)[0]
    # Make 2D grid of indices
    r = tf.range(size)
    ii, jj = tf.meshgrid(r, r, indexing='ij')
    # Take pairs of indices where the first is less or equal than the second
    m = ii < jj
    idx = tf.stack([tf.boolean_mask(ii, m), tf.boolean_mask(jj, m)], axis=1)
    # Gather result
    result_features = tf.gather(features, idx)
    result_labels= tf.gather(labels, idx)

    i = tf.constant(0)
    l = tf.Variable([])
    def condition(i, l):
        return tf.less(i, tf.shape(result_features)[0])

    def body(i, l):
        # do something here which you want to do in your loop
        # increment i

        dis = tf.nn.l2_loss(result_features[i][0] - result_features[i][1])
        similar = tf.cond(tf.equal(result_labels[i][0], result_labels[i][1]), lambda: 1.0, lambda: -1.0)
        temp = tf.maximum(0.,epsilon - similar*(theta-dis))
        l = tf.concat([l, [temp]], 0)
        return tf.add(i, 1), l

    # do the loop:
    _, list_val = tf.while_loop(condition, body, [i, l], shape_invariants=[i.get_shape(),
                                                                           tf.TensorShape([None])])
    marginal_loss = tf.reduce_sum(list_val)
    marginal_loss = tf.divide(marginal_loss, tf.cast(tf.subtract(tf.square(size),size), dtype=tf.float32))

    return marginal_loss

# CNN Architectures
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

#Overall System
def build_network(input_images, labels, ratio):
    logits, features = inference(input_images)

    with tf.name_scope('loss'):

        with tf.name_scope('marginal_loss'):
            marginal_loss = get_marginal_loss(features, labels)

        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            print(softmax_loss)
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * marginal_loss

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('MarginalLoss', marginal_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)

    return logits, features, total_loss, accuracy, softmax_loss , marginal_loss

logits, features, total_loss, accuracy, softmax_loss, marginal_loss = build_network(input_images, labels, ratio=LAMBDA)

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
# Save to logdir
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
    _, summary_str, train_acc,printed_loss, printed_marginal_loss, printed_softmax_loss = sess.run([train_op, summary_op, accuracy, total_loss, marginal_loss, softmax_loss],feed_dict={input_images: batch_images - mean_data,labels: batch_labels})
    step += 1
    print("Total_Loss: {:.6f}".format(printed_loss))
    print("Marginal_Loss: {:.6f}".format(printed_marginal_loss))
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