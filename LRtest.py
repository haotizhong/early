import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data


class LR_model:
    def __init__(self, x, y_):

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
    
    def param2vec(self, params):
        veclength = 0
        for v in params:
            dimen = 1
            for d in v.shape:
                dimen *= int(d)
            veclength += dimen
            
        vec = np.zeros(veclength)
        paramcount = 0
        for v in params:
            if len(v.shape)>1:
                v_np_vec = v.reshape(v.shape[0]*v.shape[1])
            else:
                v_np_vec = v.reshape(v.shape[0])
            vec[paramcount:paramcount+len(v_np_vec)] = v_np_vec
            paramcount += len(v_np_vec)
                   
        return vec
    
    def vec2param(self, vec, sess):
        paramcount = 0
        #TODO: convert param to tf tensors
        op_list = []
        for v in self.var_list:
            if len(v.shape) > 1:
                dim = int(v.shape[0]) * int(v.shape[1])
                param = vec[paramcount:paramcount+dim].reshape([int(v.shape[0]),int(v.shape[1])])
            else:
                dim = int(v.shape[0])
                param = vec[paramcount:paramcount+dim]           
            a_op = v.assign(param)
            op_list.append(a_op)
            paramcount += dim
        sess.run(op_list)
            
    def l2projection(self, theta0, c):
        """
        projection onto l2 ball with radius sqrt(c). i.e.:
            argmin ||u-theta0|| st ||u - center||^2 <= c
        Input:
            theta0: original variable before projection
            c: constraint upper bound, sqrt(c) is radius of l2 ball
        """
        center = self.param2vec([self.avg_weights, self.avg_bias])
        if np.shape(center) != np.shape(theta0):
            raise ValueError("dimension of theta and center must match")
        theta_proj =  c**(0.5) * (theta0-center)/max(c**(0.5), np.linalg.norm(theta0-center,2)) + center
        return theta_proj

    def init_with_avg(self, sess):  
        a_op0 = self.var_list[0].assign(self.avg_weights)
        a_op1 = self.var_list[1].assign(self.avg_bias)
        sess.run([a_op0, a_op1])
        

class data:
	def __init__(self, X_train, Y_train, X_test, Y_test):
		self.train_dats = X_train
		self.test_dats = X_test
		self.train_labels = Y_train
		self.test_labels = Y_test
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


def bootstrap(cur_data, ratio):
    rd_idx = random.sample(range(0, len(cur_data.train_dats)), int(len(cur_data.train_dats) * ratio))

    train_imgs = []
    train_labels = []
    for idx in rd_idx:
        train_imgs.append(cur_data.train_dats[idx])
        train_labels.append(cur_data.train_labels[idx])

    return data(train_imgs, train_labels, cur_data.test_dats, cur_data.test_labels)


def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_, c, as_init, sess):
    # initialize test accuracy array for each task 
    test_accs = np.zeros(num_iter/disp_freq)
    if not as_init:
        model.init_with_avg(sess)
    # train on current task
    for iter in range(num_iter):
        batch = trainset.next_batch(100)
        model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        
        if not as_init:

            stop_loss = model.get_ewc_loss(sess)
            if iter % disp_freq == 0:
                print(stop_loss)
            if stop_loss > c:
                theta0 = model.param2vec(sess.run(model.var_list))
                theta_project = model.l2projection(theta0, c)
                model.vec2param(theta_project, sess)
            	break
        
        if iter % disp_freq == 0:
            feed_dict={x: testsets.test_dats, y_: testsets.test_labels}
            test_accs[iter/disp_freq] = model.accuracy.eval(feed_dict=feed_dict)
            
    test_accs = np.array(test_accs)
    np.save('lracc' + str(c) + '.npy', test_accs)

    weights = []
    if as_init:
        	weights = sess.run(model.var_list)
    return weights


'''
READ DATA HERE
cur_data = ...
'''    
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
cur_data = data([], [], [], [])
for v in mnist.train.images:
    cur_data.train_dats.append(v)
for v in mnist.test.images:
    cur_data.test_dats.append(v)
for v in mnist.train.labels:
    cur_data.train_labels.append(v)
for v in mnist.test.labels:
    cur_data.test_labels.append(v)


feature_dim = 784
class_num = 10
num_sampling = 10
sampling_ratio = 0.1
c = 100

x = tf.placeholder(tf.float32, shape=[None, feature_dim])
y_ = tf.placeholder(tf.float32, shape=[None, class_num])

avg_weights = np.zeros([feature_dim, class_num])
avg_bias = np.zeros(class_num)

for _ in range(num_sampling):
	tmpmodel = LR_model(x, y_)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		tmp_data = bootstrap(cur_data, sampling_ratio)
		tmp_weights = train_task(tmpmodel, 800, 20, cur_data, cur_data, x, y_, c, True, sess)
		avg_weights += tmp_weights[0] / num_sampling
		avg_bias += tmp_weights[1] / num_sampling

model = LR_model(x, y_)
model.set_avg_theta(avg_weights, avg_bias)

with tf.Session() as sess:
    c = 30
    sess.run(tf.global_variables_initializer())
    train_task(model, 1600, 20, cur_data, cur_data, x, y_, c, False, sess)
