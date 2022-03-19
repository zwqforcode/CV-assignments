import numpy as np
from random import shuffle


class SVM(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=200,
            batch_size=500, verbose=False):
  
    num_train, dim = X.shape[:]
    y = y.astype(int)
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None


      batch_inx = np.random.choice(num_train,batch_size)
      X_batch = X[batch_inx,:]
      y_batch = y[batch_inx]
  

      # evaluate loss and gradient
      loss, grad = self.loss(self.W, X_batch, y_batch, reg)
      loss_history.append(loss)

      
      self.W = self.W - learning_rate * grad


      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history


  def loss(self,W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1) #(N, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W * W)
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = - np.sum(coeff_mat, axis = 1)
    dW = (X.T).dot(coeff_mat)
    dW /= num_train
    dW += reg * W

    return loss, dW


  def predict(self, X):
    y_pred = np.zeros(X.shape[0])
    score = X.dot(self.W)
    y_pred = np.argmax(score,axis=1)
    return y_pred