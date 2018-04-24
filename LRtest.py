import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import random


class LR_model:
    def __init__(self, dim):

        in_dim = int(x.get_shape()[1]) 
        out_dim = int(y_.get_shape()[1])

        self.x = x # input placeholder
        self.y_ = y_

        # simple 2-layer network
        W = weight_variable([in_dim,out_dim])
        b = bias_variable([out_dim])

        self.y = tf.nn.softmax(tf.matmul(x,W) + b)

        self.var_list = [W, b]

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

    def set_avg_theta(self, avg_weights, avg_bias):
        	self.avg_weights = avg_weights
        	self.avg_bias = avg_bias

    def get_ewc_loss(self, sess):

        stop_loss = sess.run(tf.reduce_sum(tf.square(self.var_list[0] - self.avg_weights)))
        stop_loss += sess.run(tf.reduce_sum(tf.square(self.var_list[1] - self.avg_bias)))

        return stop_loss


class data:
	def __init__(self, X_train, Y_train, X_test, Y_test):
		self.train_dats = X_train
		self.test_dats = X_test
		self.train_labels = Y_train
		self.test_labels = Y_train
		self.train_idx = 0

	def next_batch(self, num):
		batch_x = []
		batch_y = []
		for i in range(num):
			batch_x.append(self.train_dats[self.train_idx])
			batch_y.append(self.train_labels[self.train_idx])
			self.train_idx = (self.train_idx + 1) % len(self.train_dats)
		return [batch_x, batch_y]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')


def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    display.display(plt.gcf())
    display.clear_output(wait=True)


def bootstrap(cur_data, ratio):
    rd_idx = random.sample(range(0, len(cur_data.train_dats)), len(cur_data.train_dats) * ratio)

    train_imgs = []
    train_labels = []
    for idx in rd_idx:
        train_imgs.append(cur_data.train_dats[idx])
        train_labels.append(cur_data.train_labels[idx])

    return data(train_imgs, train_labels, cur_data.test_dats, cur_data.test_labels)


def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_, c, as_init):
    # initialize test accuracy array for each task 
    test_accs = np.zeros(num_iter/disp_freq)
    # train on current task
    for iter in range(num_iter):
        batch = trainset.next_batch(100)
        model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        
        if not as_init:

            stop_loss = model.get_ewc_loss(sess)
            if stop_loss > c:
                #projection
            	break
        
        if iter % disp_freq == 0:
            feed_dict={x: testsets.test_dats, y_: testsets.test_labels}
            test_accs[iter/disp_freq] = model.accuracy.eval(feed_dict=feed_dict)
            
    #test_accs = np.array(test_accs)
    #np.save('acc' + str(c) + '.npy', test_accs)

    weights = []
    if as_init:
    	weights = model.var_list
    return weights


'''
READ DATA HERE
cur_data = ...
'''

feature_dim = 784
class_num = 10
num_sampling = 10.
sampling_ratio = 0.1
c = 1

x = tf.placeholder(tf.float32, shape=[None, feature_dim])
y_ = tf.placeholder(tf.float32, shape=[None, class_num])

avg_weights = np.zeros(feature_dim, class_num)
avg_bias = np.zeros(class_num)

for _ in range(num_sampling):
	tmpmodel = LR_model(x, y_)
	with tf.Session as sess:
		sess.run(tf.global_variables_initializer())
		tmp_data = bootstrap(cur_data, sampling_ratio)
		tmp_weights = train_task(tmpmodel, 800, 20, cur_data, [cur_data], x, y_, c, True)
		avg_weights += tmp_weights[0] / num_sampling
		avg_bias += tmp_weights[1] / num_sampling

model = LR_model(x, y_)
model.set_avg_theta(avg_weights, avg_bias)

c = 1
train_task(model, 800, 20, cur_data, [cur_data], x, y_, c, False)
