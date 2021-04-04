import numpy as np
import myNN as nn
import plot_boundary_on_data 
from matplotlib import pyplot as plt
import sys
import tensorflow as tf

# load data
if(sys.argv[1] == "CIFAR10"):
    sys.argv[1] = "data/moon.npz"

data = np.load(sys.argv[1], allow_pickle=True)
X = data['X']
Y = data['Y']
K = np.unique(Y).size


# define a MLP network. First layer is a fully connected layer with 20 output
# nodes. Second layer is ReLU. Third layer is another fully connected layer with
# K nodes and the final layer is a "softmax + cross-entropy" layer. 


network = [
        [nn.fc, [np.random.rand(X.shape[1],20), np.zeros((1,20))] ],    # first layer
        [nn.relu, None],                                 # second layer
        [nn.fc, [np.random.rand(20,K), np.zeros((1,K))]],             # third layer
        [nn.softmax_cross_entropy, Y]              # fourth layer
        ]

eta= .1
num_iters = 5000

# train it
network, loss_vals = nn.train(network, X, Y, eta, num_iters)


# training finished, now replace the last layer with softmax without the
# cross-entropy loss
network[-1][0] = nn.softmax_readout

predicted_class = nn.evaluate(network, X)
print('training accuracy: %.2f' % (np.mean(predicted_class == Y)))

plot_boundary_on_data.plot(X, Y, lambda x: nn.evaluate(network, x))

fig = plt.figure()
plt.plot(loss_vals)

plt.show()
