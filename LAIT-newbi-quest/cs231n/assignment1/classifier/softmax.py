from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_class=W.shape[1]
      


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
      scores=np.exp(X[i].dot(W))
      soft_max_scores=scores/sum(scores) # 소프트멕스 score 값 구함.
      loss+=-np.log(soft_max_scores[y[i]])
      #dW=-x+soft_max_scores[j]
      for j in range(num_class):
        if j!=y[i]:
          dW[:,j]+=X[i]*soft_max_scores[j] # softmax score값 만큼 더해주고
        else:
          dW[:,y[i]]=((scores[j]-sum(scores))/sum(scores))*X[i] # 정답일 때 값만 - 붙는다. 

    loss/=num_train
    loss+=reg*np.sum(W*W)
    dW/=num_train
    dW+=2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_class=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=np.exp(X.dot(W))
    soft_max_score=np.divide(scores,np.reshape(np.sum(scores,axis=1),(-1,1)))
    loss+=sum(-np.log(soft_max_score)[np.arange(num_train),y[np.arange(num_train)]])
    loss/=num_train
    loss+=reg*np.sum(W*W)
    dW


    
    soft_max_scores=scores

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
