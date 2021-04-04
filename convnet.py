import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import keras 
from tensorflow import keras


# data = load CIFAR10 dataset
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# parameters
batch_size = 16
display_step = 10
learning_rate = .01
iters = 1000

# convert labels to one-hot vectors
# to-do
NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[random_index, :])
        ax.set_title(cifar10_classes[y_train[random_index, 0]])
plt.show()
x_train2 = (x_train / 255) - 0.5
x_test2 = (x_test / 255) - 0.5
# convert class labels to one-hot encoded, should have shape (?, NUM_CLASSES)
y_train2 = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test2 = keras.utils.to_categorical(y_test, NUM_CLASSES)
# define variables
#x =  to-do
# y = to-do

# conv2d function for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x_out = tf.nn.relu(x)
    return x_out

# maxpool2d function for simplicity
def maxpool2d(x, k=2):
    # MaxPool2D
    x_out = tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return x_out

# define CNN
def conv_net(x):  

    weights = {
    'wc1': tf.Variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.Variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.Variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.Variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.Variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
    }
    biases = {
    'bc1': tf.Variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.Variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.Variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.Variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.Variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
    }
    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
def visualizaWeights(weights):


    for wname, w in weights.items():
        # Take only conv weights
        if wname.startswith('conv'): 
            
            plt.figure()

            nFilter = w.shape[3]
            k = w.shape[0]

            for i in range(nFilter):
                #img =  to-do
                
                plt.subplot(1,nFilter,i)
                plt.cla()
                
                plt.imshow(img)
                
                frame = plt.gca()
                frame.axes.get_xaxis().set_ticks([])
                frame.axes.get_yaxis().set_ticks([])
                
            plt.title('Filters:' + wname[1:])
            plt.savefig('filters_'+ wname[1:] + '.png')
                
# construct model
pred = conv_net(x_train2)

# Define loss and optimizer
#cost = to-do
#optimizer =  to-do
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred =  to-do
# accuracy = to-do
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables

loss_vals = []
step = 0 
init = tf.global_variables_initializer()
for epoch in range(iters):
    # get batch        
    # step = step + 1 
    
    # forward pass
      
    if step % display_step == 0:
        # Calculate  loss and accuracy
        #  loss = to-do
        #
        acc = to-do
        print("Iteration " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    loss_vals.append(loss)
    
    # run optimization 
    # backward pass
        
print("Optimization Finished!")

# Calculate accuracy for test images
# test_acc = to-do
print("Testing Accuracy:", test_acc)   
        
        
fig = plt.figure()
plt.plot(loss_vals)
plt.show()

#weights = get network weights
visualizaWeights(weights)