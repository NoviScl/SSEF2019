## Codes adapted from: https://github.com/PotatoThanh/ADMM-NeuralNetworks/tree/master/python

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

class ADMM_NN(object):
    def __init__(self, n_inputs, n_hiddens, n_outputs, n_batches):
        """
        Initialize variables for NN.
        Not sure how initialization affects the performance.
        The data should be in columns.
        for example, the input size of MNIST data should be (28x28, *) instead of (*, 28x28).
        Ignored bias terms.
        :param n_inputs: Number of inputs.
        :param n_hiddens: Number of hidden units.
        :param n_outputs: Number of outputs
        :param n_batches: Number of data sample that you want to train
        """
        self.a0 = np.zeros((n_inputs, n_batches))

        self.w1 = np.zeros((n_hiddens, n_inputs))
        self.w2 = np.zeros((n_hiddens, n_hiddens))
        self.w3 = np.zeros((n_outputs, n_hiddens))

        self.z1 = np.random.rand(n_hiddens, n_batches)
        self.a1 = np.random.rand(n_hiddens, n_batches)

        self.z2 = np.random.rand(n_hiddens, n_batches)
        self.a2 = np.random.rand(n_hiddens, n_batches)

        self.z3 = np.random.rand((n_outputs, n_batches))

        self.lambda_lagrange = np.ones((n_outputs, n_batches))

    def _relu(self, x):
        return tf.maximum(0.0, x)

    def _weight_update(self, layer_output, activation_input):
        """
        Consider it now the minimization of the problem with respect to W_l.
        For each layer l, the optimal solution minimizes ||z_l - W_l a_l-1||^2. This is simply
        a least square problem, and the solution is given by W_l = z_l p_l-1, where p_l-1
        represents the pseudo-inverse of the rectangular activation matrix a_l-1.
        :param layer_output: output matrix (z_l)
        :param activation_input: activation matrix l-1  (a_l-1)
        :return: weight matrix
        """
        pinv = np.linalg.pinv(activation_input)
        weight_matrix = tf.matmul(tf.cast(layer_output, tf.float32), tf.cast(pinv, tf.float32))
        return weight_matrix 

    def _activation_update(self, next_weight, next_layer_output, layer_nl_output, beta, gamma):
        """
        Minimization for a_l is a simple least squares problem similar to the weight update.
        However, in this case the matrix appears in two penalty terms in the problem, and so
        we must minimize:
            beta ||z_l+1 - W_l+1 a_l||^2 + gamma ||a_l - h(z_l)||^2
        :param next_weight:  weight matrix l+1 (w_l+1)
        :param next_layer_output: output matrix l+1 (z_l+1)
        :param layer_nl_output: activate output matrix h(z) (h(z_l))
        :param beta: value of beta
        :param gamma: value of gamma
        :return: activation matrix
        """
        layer_nl_output = self._relu(layer_nl_output)
        
        next_weight = tf.cast(next_weight, tf.float64)
        m1 = beta*tf.matmul(tf.cast(tf.matrix_transpose(next_weight), tf.float64), next_weight)
        m2 = tf.scalar_mul(gamma, tf.eye(tf.cast(m1.get_shape()[0], tf.int32)))
        av = tf.matrix_inverse(tf.cast(m1, tf.float32) + tf.cast(m2, tf.float32))

        m3 = beta*tf.matmul(tf.matrix_transpose(next_weight), tf.cast(next_layer_output, tf.float64))
        m4 = gamma*layer_nl_output 
        af = tf.cast(m3, tf.float32) + tf.cast(m4, tf.float32)

        return tf.matmul(av, af)


    def _argminz(self, a, w, a_in, z_in, beta, gamma):
        """
        This problem is non-convex and non-quadratic (because of the non-linear term h).
        Fortunately, because the non-linearity h works entry-wise on its argument, the entries
        in z_l are decoupled. This is particularly easy when h is piecewise linear, as it can
        be solved in closed form; common piecewise linear choices for h include rectified
        linear units (ReLUs), that is used here, and non-differentiable sigmoid functions.
        :param a: activation matrix (a_l)
        :param w:  weight matrix (w_l)
        :param a_in: activation matrix l-1 (a_l-1)
        :param beta: value of beta
        :param gamma: value of gamma
        :param z_in: z_l (matrix)
        :return: output matrix
        """
        m = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        #note that z can be either postive or negative,
        #we need to compute for both possibilities and take the min
        sol1 = (gamma*a + beta*m) / (gamma + beta)  #if z>=0
        sol2 = m  #if z<0 
        
        sol1 = np.array(sol1)
        sol2 = np.array(sol2)
        z_in = np.array(z_in)
        z = np.zeros_like(z_in)
        
        z[z_in>=0.] = sol1[z_in>=0.]
        z[z_in<0.] = sol2[z_in<0.]

        return z 

    def _argminlastz(self, targets, eps, w, a_in, beta):
        """
        Minimization of the last output matrix.
        Using square error as loss term here.
        Treat lagrange as an element-wise product and find min of quadratic function.
        target(y), lambda, z_L, all same dimension
        :param targets: target matrix (equal dimensions of z) (y)
        :param eps: lagrange multiplier matrix (equal dimensions of z) (lambda)
        :param w: weight matrix (w_l)
        :param a_in: activation matrix l-1 (a_l-1)
        :param beta: value of beta
        :return: output matrix last layer
        """
        m = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        z = (targets - eps/2 +beta*m) / (1+beta)
        return z 

    def _lambda_update(self, zl, w, a_in, beta):
        """
        Lagrange multiplier update.
        :param zl: output matrix last layer (z_L)
        :param w: weight matrix last layer (w_L)
        :param a_in: activation matrix l-1 (a_L-1)
        :param beta: value of beta
        :return: lagrange update
        """
        mpt = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        lambda_up = beta*(zl - mpt)

        return self.lambda_lagrange + lambda_up 

    def feed_forward(self, inputs):
        """
        Calculate feed forward pass for neural network
        :param inputs: inputs features
        :return: value of prediction
        """
        outputs = self._relu(tf.matmul(self.w1, inputs))
        outputs = self._relu(tf.matmul(self.w2, outputs))
        # no activation for final layer
        outputs = tf.matmul(self.w3, outputs)
        return outputs



    def fit(self, inputs, labels, beta, gamma):
        """
        Training ADMM Neural Network by minimizing sub-problems
        :param inputs: input of training data samples
        :param outputs: label of training data samples
        :param epochs: number of epochs
        :param beta: value of beta
        :param gamma: value of gamma
        :return: loss value
        """
        self.a0 = inputs 

        # Input layer 
        self.w1 = self._weight_update(self.z1, self.a0)
        self.a1 = self._activation_update(self.w2, self.z2, self.z1, beta, gamma)
        self.z1 = self._argminz(self.a1, self.w1, self.a0, self.z1, beta, gamma) 

        # Hidden layer (use loop if many layers)
        self.w2 = self._weight_update(self.z2, self.a1)
        self.a2 = self._activation_update(self.w3, self.z3, self.z2, beta, gamma)
        self.z2 = self._argminz(self.a2, self.w2, self.a1, self.z2, beta, gamma)

        # Output layer 
        self.w3 = self._weight_update(self.z3, self.a2)
        self.z3 = self._argminlastz(labels, self.lambda_lagrange, self.w3, self.a2, beta)
        self.lambda_lagrange = self._lambda_update(self.z3, self.w3, self.a2, beta)

        loss, accuracy = self.evaluate(inputs, labels)
        return loss, accuracy 


    def evaluate(self, inputs, labels, isCategories=True):
        """
        Calculate loss and accuracy (only classification)
        :param inputs: inputs data
        :param outputs: ground truth
        :param isCategrories: classification or not
        :return: loss and accuracy (only classification)
        """
        forward = self.feed_forward(inputs)
        loss = tf.reduce_mean(tf.square(forward - labels))

        if isCategories:
            accuracy = tf.equal(tf.argmax(labels, axis=0), tf.argmax(forward, axis=0))
            accuracy = tf.reduce_sum(tf.cast(accuracy, tf.int32)) / accuracy.get_shape()[0]
        else:
            # for regression, no so-called accuracy
            accuracy = loss 

        return loss, accuracy

    def warming(self, inputs, labels, epochs, beta, gamma):
        """
        Warming ADMM Neural Network by minimizing sub-problems without update lambda for several iterations
        :param inputs: input of training data samples
        :param outputs: label of training data samples
        :param epochs: number of epochs
        :param beta: value of beta
        :param gamma: value of gamma
        :return:
        """
        self.a0 = inputs 
        for i in range(epochs):
            print("------ Warming: {:d} ------".format(i))

            #Input layer 
            self.w1 = self._weight_update(self.z1, self.a0)
            self.a1 = self._activation_update(self.w2, self.z2, self.z1, beta, gamma)
            self.z1 = self._argminz(self.a1, self.w1, self.a0, self.z1, beta, gamma) 

            #Hidden layer
            self.w2 = self._weight_update(self.z2, self.a1)
            self.a2 = self._activation_update(self.w3, self.z3, self.z2, beta, gamma)
            self.z2 = self._argminz(self.a2, self.w2, self.a1, self.z2, beta, gamma)

            # Output layer
            self.w3 = self._weight_update(self.z3, self.a2)
            self.z3 = self._argminlastz(labels, self.lambda_lagrange, self.w3, self.a2, beta)


    def drawcurve(self, train_, valid_, legend_1, legend_2):
        acc_train = np.array(train_).flatten()
        acc_test = np.array(valid_).flatten()

        plt.figure()
        plt.plot(acc_train)
        plt.plot(acc_test)

        plt.legend([legend_1, legend_2], loc="upper left")
        plt.draw()
        plt.pause(0.001) 

























