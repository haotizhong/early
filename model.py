import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

# variable initialization functions
def weight_variable(shape, weight_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = weight_name)

def bias_variable(shape, weight_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = weight_name)

class Model:
    def __init__(self, learningrate, x, y_):

        self.in_dim = int(x.get_shape()[1]) # 784 for MNIST
        self.out_dim = int(y_.get_shape()[1]) # 10 for MNIST

        self.x = x # input placeholder
        self.y_ = y_

        # simple 2-layer network
        W1 = weight_variable([self.in_dim,50], 'W1')
        b1 = bias_variable([50], 'b1')

        W2 = weight_variable([50,self.out_dim], 'W2')
        b2 = bias_variable([self.out_dim], 'b2')

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
        self.y = tf.matmul(h1,W2) + b2 # output layer

        self.var_list = [W1, b1, W2, b2]
        
        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss(learningrate)

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.F_accum = []
        
    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter
        # initialize Fisher information for most recent task
        
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            print("fisher " + str(i))
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())
    
                
    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self,learningrate):
        self.train_step = tf.train.GradientDescentOptimizer(learningrate).minimize(self.cross_entropy)

    def update_ewc_loss(self, learningrate, lam):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            self.ewc_loss += lam * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(learningrate).minimize(self.ewc_loss)

    def get_ewc_loss(self, sess):
        # compute elastic weight term
        ewc_loss = 0
        for v in range(len(self.var_list)):
            ewc_loss += np.sum(self.F_accum[v].astype(np.float32) * (self.var_list[v].eval() - self.star_vars[v])**2)
 
        return ewc_loss
    
    def param2vec(self):
        
        veclength = 0
        for v in self.var_list:
            dimen = 1
            for d in v.shape:
                dimen *= int(d)
            veclength += dimen
            
        vec = np.zeros(veclength)
        paramcount = 0
        for v in self.var_list:
            v_np = v.eval()
            if len(v_np.shape)>1:
                v_np_vec = v_np.reshape(v_np.shape[0]*v_np.shape[1])
            else:
                v_np_vec = v_np.reshape(v_np.shape[0])
            vec[paramcount:paramcount+len(v_np_vec)] = v_np_vec
            paramcount += len(v_np_vec)
                   
        return vec 


    def vec2param(self, vec, sess):
        paramcount = 0
        #TODO: convert param to tf tensors
        for v in self.var_list:
            if len(v.shape) > 1:
                dim = int(v.shape[0]) * int(v.shape[1])
                param = vec[paramcount:paramcount+dim].reshape([int(v.shape[0]),int(v.shape[1])])
            else:
                dim = int(v.shape[0])
                param = vec[paramcount:paramcount+dim]           
            v.assign(param)
            sess.run(v)
            paramcount += dim
            
            
 
    
            
            
            
        


                    
                
        
        
        
        
