import numpy as np
import matplotlib.pyplot as plt
import myNN as nn
import sys
import plot_boundary_on_data

# load data
data = np.load(sys.argv[1])
X = data['X']
Y = data['Y']
K = np.unique(Y).size


# define a 2-layer softmax network. First layer is a fully connected layer with
# K output nodes. The second layer is a "softmax + cross-entropy" layer. 
network = [
        [nn.fc, [np.random.rand(X.shape[1],K), np.random.rand(1,K)]], # first layer
        [nn.softmax_cross_entropy, Y]          # second layer
        ]

# train it
network, loss_vals = nn.train(network, X, Y, .1, 1000)


# training finished, now replace the last layer with softmax without the
# cross-entropy loss
network[-1][0] = nn.softmax_readout

predicted_class = nn.evaluate(network, X)
print('training accuracy: %.2f' % (np.mean(predicted_class == Y)))

plot_boundary_on_data.plot(X, Y, lambda x: nn.evaluate(network, x))

fig = plt.figure()
plt.plot(loss_vals)

plt.show()
