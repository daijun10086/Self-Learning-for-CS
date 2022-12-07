import numpy as np
import matplotlib.pyplot as plt

# Generate some data
N = 100  # the number of points per class
D = 2  # dimensionality
K = 3  # number of classes
h = 100  # size of hidden layer
X = np.zeros((N * K, D))  # data matrix
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
    y[ix] = j
# Let visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# initialize the parameters
W1 = 0.01 * np.random.randn(D, h)
b1 = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

reg = 1e-3  # set the reg efficient
step_size = 1e-0  # set updating step
# Train a Linear Classifier
# Gradient descent loop
for i in range(10000):
    # compute the scores for a linear classifier
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # using ReLU non-linear activation
    scores = np.dot(hidden_layer, W2) + b2

    # compute the cross-entropy loss
    num_examples = X.shape[0]
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalized the probabilities
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    # compute the loss: average cross-entropy loss and regularization
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))

    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dW2 += reg * W2

    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0
    dW1 = np.dot(X.T, dhidden)
    dW1 += reg * W1
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # perform a parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f" % (np.mean(predicted_class == y)))
