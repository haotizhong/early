import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
from model import Model
from optroutine import ellipsoidprojection

def mnist_imshow(img):
    plt.imshow(img.reshape([28,28]), cmap="gray")
    plt.axis('off')

# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist):
    perm_inds = np.array(range(mnist.train.images.shape[1]))
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2


def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
# train/compare vanilla sgd and ewc
def train_task(model, num_iter, disp_freq, trainset, testsets, x, y_, learningrate ,c, lams=[0]):

    for l in range(len(lams)):
        model.restore(sess) # reassign optimal weights from previous training session
        if(lams[l] == 0) or not hasattr(model, "star_vars"):
            model.set_vanilla_loss(learningrate)
        else:
            model.update_ewc_loss(learningrate,lams[l])
        # initialize test accuracy array for each task 
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(int(num_iter/disp_freq)))
        # train on current task
        for iter in range(num_iter):
            print(iter)
            batch = trainset.train.next_batch(100)
            model.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            if hasattr(model, "star_vars"):
                stop_loss = model.get_ewc_loss(sess)
                print(stop_loss)
                if stop_loss > c:
                    
                    #ellipsoidprojection(theta0,center,weights,c)
                    #projection
                    break

            if iter % disp_freq == 0:
                plt.subplot(1, len(lams), l+1)
                plots = []
                colors = ['r', 'b', 'g']
                for task in range(len(testsets)):
                    feed_dict={x: testsets[task].test.images, y_: testsets[task].test.labels}
                    test_accs[task][int(iter/disp_freq)] = model.accuracy.eval(feed_dict=feed_dict)
                    ch = chr(ord('A') + task)
                    plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[task][:int(iter/disp_freq+1)], colors[task], label="task " + ch)
                    plots.append(plot_h)
                plot_test_acc(plots)
                if l == 0: 
                    plt.title("vanilla sgd")
                else:
                    plt.title("ewc")
                plt.gcf().set_size_inches(len(lams)*5, 3.5)

    return test_accs
# end train_task
    
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
model = Model(0.1,x, y_)
sess.run(tf.global_variables_initializer())


c = 0.2
accu1 = train_task(model, 3000, 20, mnist, [mnist], x, y_, 0.1, c, lams=[0])
model.compute_fisher(mnist.validation.images, sess, num_samples=500, plot_diffs=False) # use valida
mnist2 = permute_mnist(mnist)
model.star()
#
accu2 = train_task(model, 3000, 20, mnist2, [mnist, mnist2], x, y_, 0.1, c, lams=[1])
#
model.compute_fisher(mnist2.validation.images, sess, num_samples=500, plot_diffs=False)
mnist3 = permute_mnist(mnist)
model.star()
#
accu3 = train_task(model, 3000, 20, mnist3, [mnist, mnist2, mnist3], x, y_, 0.1, c, lams=[1])
