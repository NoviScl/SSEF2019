## currently it's purely based on numpy
## you can use Numba to accelerate
import numpy as np 
import matplotlib.pyplot as plt 

def relu(x):
    return np.maximum(0., x)

# NOTE: fit and feed forward are two seperate things
# fit will use the trained parameters
# pseudo inverse: matrix dimension will transpose
# data need to be in columns
class RNN(object):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_BATCH, SEQ_LEN):
        # N_BATCH: number of examples in one batch
        # initialize weight matrix for input, hidden, output matrix
        # these weights are shared
        self.wi = np.random.normal(0, 1, [HIDDEN_DIM, INPUT_DIM])
        self.wh = np.random.normal(0, 1, [HIDDEN_DIM, HIDDEN_DIM])
        self.wo = np.random.normal(0, 1, [OUTPUT_DIM, HIDDEN_DIM])

        self.h = list()
        self.u = list()
        self.v = list()

        # intialize variables in RNN
        # NOTE: initial h is always initialized as zero matrix and is not trained!
        for t in range(SEQ_LEN):
            # could try initialize h to all zeros 
            self.h.append(np.random.normal(0, 1, [HIDDEN_DIM, N_BATCH]))
            self.u.append(np.random.normal(0, 1, [HIDDEN_DIM, N_BATCH]))
            self.v.append(np.random.normal(0, 1, [HIDDEN_DIM, N_BATCH]))

        self.yT =  np.random.normal(0, 1, [OUTPUT_DIM, N_BATCH])

        # list to store the loss and accuracy information
        self.errs = list()
        self.accs = list()

        # same shape as the the final otuput
        self.lambda_lagrange = np.ones((OUTPUT_DIM, N_BATCH))

    def wi_update(self, ut, xt):
        pinv = np.linalg.pinv(xt)
        wi = np.dot(ut, pinv)
        return wi 

    def wh_update(self, vt, hprev):
        pinv = np.linalg.pinv(hprev)
        wh = np.dot(vt, pinv)
        return wh 

    def wo_update(self, yT, hT):
        pinv = np.linalg.pinv(hT)
        wo = np.dot(yT, pinv)
        return wo 

    def ut_update(self, ut, vt, wi, xt, ht, alpha, gamma):
        u_v = ut + vt 
        new_ut = np.zeros_like(ut)
        sol1 = np.dot(wi, xt)  # u_v < 0
        sol2 = (alpha*ht + gamma*np.dot(wi, xt) - alpha*vt)/(gamma+alpha)

        new_ut[u_v>=0.] = sol2[u_v>=0.]
        new_ut[u_v<0.] = sol1[u_v<0.]

        return new_ut 

    def vt_update(self, ut, vt, wh, hprev, ht, omega, alpha):
        u_v = ut + vt 
        new_vt = np.zeros_like(vt)
        sol1 = np.dot(wh, hprev) # u_v < 0 
        sol2 = (omega*np.dot(wh, hprev) - alpha*ut + alpha*ht)/(omega+alpha)

        new_vt[u_v>=0.] = sol2[u_v>=0.]
        new_vt[u_v<0] = sol1[u_v<0]

        return new_vt

    def ht_update(self, omega, vnext, wh, alpha, ut, vt):
        parta = omega*np.dot(wh, vnext) + alpha*tanh(ut + vt)
        partb = omega*np.dot(wh.T, wh) + alpha*np.eye(wh.shape[1])
        return np.dot(np.linalg.pinv(partb), parta)

    # update last output
    def hT_update(self, yT, wo):
        hT = np.dot(np.linalg.pinv(wo), yT)
        return hT 

    # necessary becoz yT is used in wo update
    # target is in one-hot format
    # target: (OUTPUT_DIM, N_BATCH)
    def yT_update(self, target, beta, wo, hT, lambda_lagrange):
        yT = (target + beta*np.dot(wo, hT) - lambda_lagrange/2)/(1+beta)
        return yT 

    def lambda_update(self, beta, yT, wo, hT):
        lambda_up = beta*(yT - np.dot(wo, hT))
        return self.lambda_lagrange + lambda_up 

    # input shape: (N_BATCH, seq_len, INPUT_DIM)
    # many-to-one
    def feed_forward(self, inputs):
        seq_len = inputs.shape[1]

        hidden = np.zeros((HIDDEN_DIM, N_BATCH))

        for t in range(seq_len):
            X = inputs[:, t, :]  #(INPUT_DIM, N_BATCH)
            hidden = relu(np.dot(self.wi, X) + np.dot(self.wh, hidden))

        output = np.dot(self.wo, hidden)
        return output 
        # shape: (OUTPUT_DIM, N_BATCH)

    def fit(self, inputs, labels, alpha, beta, gamma, omega):
        # inputs: (INPUT_DIM, SEQ_LEN, N_BATCH)
        seq_len = inputs.shape[1]

        for t in range(seq_len):
            xt = inputs[:, t, :]
            self.wi = self.wi_update(self.u[t], xt)
            self.u[t] = self.ut_update(self.u[t], self.v[t], self.wi, xt, self.h[t], alpha, gamma)
            if t>0:
                self.wh = self.wh_update(self.v[t], self.h[t-1])
                self.v[t] = self.vt_update(self.u[t], self.v[t], self.wh, self.h[t-1], self.h[t], omega, alpha)
            if t < seq_len-1:
                self.h[t] = self.ht_update(omega, self.v[t+1], self.wh, alpha, self.u[t], self.v[t])
            else:
                self.h[t] = self.hT_update(self.yT, self.wo)
        
        self.wo = self.wo_update(self.yT, self.h[-1])
        self.yT = self.yT_update(labels, beta, self.wo, self.h[-1], self.lambda_lagrange)
        self.lambda_lagrange = self.lambda_update(beta, self.yT, self.wo, self.h[-1])

        ## add accuracy to evaluate function if needed
        loss = self.evaluate(inputs, labels)
        return loss 


    def warming(self, inputs, labels, alpha, beta, gamma, omega, epochs):
        # inputs: (INPUT_DIM, SEQ_LEN, N_BATCH)
        seq_len = inputs.shape[1]
        for ep in range(epochs):
            print ("------ Warming: {:d} ------".format(ep))
            for t in range(seq_len):
                xt = inputs[:, t, :]
                self.wi = self.wi_update(self.u[t], xt)
                self.u[t] = self.ut_update(self.u[t], self.v[t], self.wi, xt, self.h[t], alpha, gamma)
                if t>0:
                    self.wh = self.wh_update(self.v[t], self.h[t-1])
                    self.v[t] = self.vt_update(self.u[t], self.v[t], self.wh, self.h[t-1], self.h[t], omega, alpha)
                if t < seq_len-1:
                    self.h[t] = self.ht_update(omega, self.v[t+1], self.wh, alpha, self.u[t], self.v[t])
                else:
                    self.h[t] = self.hT_update(self.yT, self.wo)
            
            self.wo = self.wo_update(self.yT, self.h[-1])
            self.yT = self.yT_update(labels, beta, self.wo, self.h[-1], self.lambda_lagrange)


    def evaluate(self, inputs, labels):
        # inputs: (input_dim, seq_len, N_BATCH)
        # labels: (output_dim, N_BATCH)
        preds = self.feed_forwaed(inputs)
        # (OUTPUT_DIM, N_BATCH)
        labels = np.asarray(labels)

        loss = np.mean(np.square(np.substract(preds, labels)))
        return loss 

    

        






















