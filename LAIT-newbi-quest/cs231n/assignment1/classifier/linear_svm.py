from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]] # 정답일때 yhat 값
        for j in range(num_classes):
            if j == y[i]: # inference 가 맞는 결과에는 loss포함안함
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0: 
              dW[:,y[i]]-=X[i,] # 오답중 y[i]정답일 때의 차이는 -x
              dW[:,j]+=X[i,:] # 오답중 오답일 때의 차이는 x
              loss += margin
    dW/=num_train
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW+=2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=X.dot(W)
    correct_class_score = scores[np.arange(num_train), y] # y=yhat일때 score
    margins = np.maximum(0, scores - correct_class_score.reshape(-1,1)  + 1) # 각 score값에서 correct-class_score를 빼야한다. 
    margins[np.arange(num_train), y] = 0 # margin 구하기 완료

    loss=np.sum(margins)/num_train +reg*np.sum(W*W) # loss 구하기 완료
    # margin은 >0일 때 hinge 값들이 모두 저장되어있다.
    X_m = np.zeros(margins.shape) # margin shape
    X_m[margins > 0] = 1 #  margin>0이면 1
    count = np.sum(X_m, axis=1) # X_m에서 1인 것들 모두 합
    X_m[np.arange(num_train), y] = -count # 
    dW = X.T.dot(X_m)
    dW /= num_train
    dW+=2*W*reg # regularization
    #dW += np.multiply(W, reg)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
