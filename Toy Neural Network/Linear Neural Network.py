import numpy as np
import matplotlib.pyplot as plt

# Generate some data
N = 100  # the number of points per class
D = 2  # dimensionality
K = 3  # number of classes
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
W = 0.1 * np.random.randn(D, K)
b = np.zeros((1, K))

reg = 0.001  # set the reg efficient
step_size = 1  # set updating step
# Train a Linear Classifier
# Gradient descent loop
for i in range(200):
    # compute the scores for a linear classifier
    scores = np.dot(X, W) + b

    # compute the cross-entropy loss
    num_examples = X.shape[0]
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalized the probabilities
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    # compute the loss: average cross-entropy loss and regularization
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))

    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f" % (np.mean(predicted_class == y)))
