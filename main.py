__author__ = 'Kuan'

from logistic_sgd import load_data
from mlp import MLP
import numpy as np
import theano.tensor as T

def main(dataset = 'mnist.pkl.gz',no_model = None, no_hid = None):
    # initiate hyper-parameters
    if no_model == None:
        no_model = 10
    if no_hid == None:
        no_hid = 5

    # data loading
    print 'loading data ...'
    datasets = load_data(dataset)
    x_tr, y_tr = datasets[0]
    x_va, y_va = datasets[1]
    x_te, y_te = datasets[2]

    x = T.matrix('x')
    y = T.ivector('y')

    no_train = x_tr.get_value(borrow=True).shape[0]
    dim = x_tr.get_value(borrow=True).shape[1]
    K = y_tr.eval().max() # 9 for mnsit

    # global variables
    weight = np.tile(1.0/no_train,(no_train,)) # weights for each instances
    alpha = np.zeros((no_model,))
    # model
    nets = []
    rng = np.random.RandomState(1234)

    for i in xrange(no_model):
        net = MLP(
            rng=rng,
            input=x,
            n_in=dim,
            n_hidden=no_hid,
            n_out=K+1
        )
        print('adding network %i into nets list')%(i)
        nets.append(net)

    # train model
    for i in xrange(no_model):
        # train model based on current weights
        # make prediction, compute weighted error/alpha
        # update weights

    return

def train_mlp():







if __name__=='__main__':
    main()