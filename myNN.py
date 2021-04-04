import numpy as np
import matplotlib.pyplot as plt

def fc(X, params):
    # Fully-connected layer
    #
    # Inputs:
    #   X: N-by-D dimensional numpy array. N: number of examples, D: number of
    #       features.
    #   params: a list containing two items: W and b which are explained below.
    #   W: D-by-K dimensional numpy array storing the weights of the fully
    #       connected layer. D is the dimensionality of the input layer, and K
    #       is the number of neurons in the output layer.
    #   b: 1-by-K dimensional numpy array storing the bias terms.
    #
    # Outputs:
    #   This function must return a Python dictionary with the following fields:
    #       out, dx, dw and db. 
    #   out: Output of the fully connected layer, which is N-by-K numpy array.
    #   dx: Derivative of the output w.r.t. the input. D-by-K dimensional.
    #   dw: Derivative of the output w.r.t. the weights. N-by-D dimensional.
    #   db: Derivative of the output w.r.t. the biases. N-by-1 dimensional.

    W,b = params
    withoutBias = X.dot(W)
    out = withoutBias + b
    dw = X.T
    dx = W.T
    db = np.ones((X.shape[0],1))
    return {'out':out, 'dx':dx, 'dw':dw, 'db':db}

def relu(X, ignored):
    # Rectified linear unit layer. 
    #
    # This layer receives two inputs. The second input is ignored. The first
    # input, X, is a M-by-N array. 
    #
    # Output is a Python dictionary with fields out and dx. 
    # out: Output of ReLU. M-by-N dimensional. 
    # dx: Derivative of the output w.r.t. the input. 

    out = np.maximum(X,0)
    dx = np.array(X, copy = True)
    dx[X > 0] = 1
    dx[X <= 0] = 0

    return {'out':out, 'dx':dx}

def softmax_cross_entropy(X, Y):
    # Softmax + cross-entropy layer. This layer is used when training the
    # network. 
    #
    # This layer receives two inputs: X and Y. 
    # X: classification scores output by the model. N-by-K array. N is the
    #   number of examples. K is the number of classes. 
    # Y: ground-truth labels, i.e. class indices, for the examples. N-by-1
    #   array.
    #
    # Output is a Python dictionary with fields out and dx. 
    # out: Output of cross-entropy. A scalar. 
    # dx: Derivative of the output w.r.t. the input. 

    # This implementation is given for you. 


    # Softmax
    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    probs = e_x / np.sum(e_x,axis=1, keepdims=True)

    N = X.shape[0]
    # compute loss
    out = -np.sum(np.log(probs[np.arange(N), Y])) / N
    # compute dx
    dx = probs.copy()
    dx[np.arange(N), Y] = dx[np.arange(N), Y] - 1
    dx = dx/N
    return {'out':out, 'dx':dx}

def softmax_readout(X, ignored):
    # Softmax layer (without cross-entropy). This layer is used when
    # evaluating/testing the trained network. 
    #
    # This layer receives two inputs. The second argument is ignored. 
    # X: classification scores output by the model. N-by-K array. N is the
    #   number of examples. K is the number of classes. 
    #
    # Output is a Python dictionary with fields out and dx. 
    # out: Output of softmax. N-by-K array.
    # dx: Derivative of the output w.r.t. the input. 

    # This implementation is given for you. 
    # Softmax

    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))

    probs = e_x / np.sum(e_x,axis=1, keepdims=True)

    return {'out':np.argmax(probs, axis=1)}


def evaluate(network, X):
    # Evaluates the given network on the given data X. Implements the forward
    # pass. 

    output_of_previous_layer = X

    for f in network:
        y = f[0](output_of_previous_layer, f[1])
        output_of_previous_layer = y['out']

    return y['out']



def train(network, X, Y, eta=.1, num_iters = 5000):
    # Trains the network on the given dataset. 
    # 
    # Inputs: 
    #   network: initial network. A network is a Python list. Each element of
    #       the list -- which is also a list --  specifies a layer in the
    #       network. Each layer is a two-element list. First element is the
    #       processing function of the layer, e.g. fc, relu, softmax, etc. By
    #       default, each layer takes the output of the previous layer as input.
    #       A layer can take an additional input which is specified in the
    #       second element of the list that defines the layer.
    #   X: N-by-D array. N is the number of examples, D: number of features. 
    #
    # Outputs are network and loss_vals. 
    #   network: The trained network. 
    #   loss_vals: Array of loss values over iterations.

    loss_vals = [];
    for iter in range(num_iters):
        output_of_previous_layer = X;

        # forward pass
        out_list = []
        for f in network:
            y = f[0](output_of_previous_layer, f[1])
            out_list.append(y)
            output_of_previous_layer = y['out']

        loss_vals.append(y['out'])

        # backward pass
        dout = out_list[-1]['dx'] # At each layer, dout is the derivative of the
                                  # training loss w.r.t. the input of that layer
        for i in range(len(network)-2,-1,-1):
            if network[i][0]==fc:
                # this is a FC layer, so we should update its parameters. FC
                # layer has two parameters W and b. W is at network[i][1][0] and
                # b is at network[i][1][1]. Write gradient descent update rules
                # for them. Use "eta" as the learning rate.


                grad_w = np.dot(out_list[i]['dw'],dout)
                grad_b = np.dot(out_list[i]['db'].T,dout)

                network[i][1][0] = network[i][1][0] - eta * grad_w
                network[i][1][1] = network[i][1][1] - eta * grad_b

                # Propagate gradient by updating dout. dout is the derivative of
                # the loss w.r.t. FC's input. 
                dout = np.dot(dout,out_list[i]['dx'])



            elif network[i][0]==relu:
                # RELU layer doesn't have any parameters, so nothing to update

                # propagate gradient
                dout = np.multiply(dout,out_list[i]['dx'])


    return network, loss_vals