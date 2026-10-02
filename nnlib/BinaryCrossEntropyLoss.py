import numpy as np
from .PLayer import PLayer

class BinaryCrossEntropyLoss(PLayer):
    """
    Binary cross-entropy (BCE) loss layer, used for binary classification such as the XOR problem.
    Measures how far predicted probabilities are from the true 0/1 labels:
        BCE(p, y) = -mean( y * log(p) + (1 - y) * log(1 - p) )
    The loss is near 0 when the prediction matches the label and grows quickly when the model is
    confidently wrong. It expects probabilities in (0, 1), so it is placed after a sigmoid layer.
    This layer has no learnable parameters.
    """

    def __init__(self, eps=1e-12):
        """
        Initialize the BinaryCrossEntropyLoss layer.

        Parameters:
            eps (float): Small constant used to clip predictions into [eps, 1 - eps],
                         so that log(0) and division by zero never happen.

        Attributes:
            y_pred (np.ndarray): Clipped predictions from the forward pass, reused in backward.
            y_true (np.ndarray): Target labels from the forward pass, reused in backward.
        """
        super().__init__(name="BinaryCrossEntropyLoss")
        self.eps = eps
        self.y_pred = None
        self.y_true = None



    def forward(self, y_pred, y_true):
        """
        Perform the forward pass: compute the mean binary cross-entropy loss.

        Parameters:
            y_pred (np.ndarray): Predicted probabilities of shape (n_samples, 1),
                                 typically the output of a sigmoid layer.
            y_true (np.ndarray): True labels (0 or 1). Reshaped to match y_pred,
                                 so shape (n_samples,) or (n_samples, 1) both work.

        Returns:
            float: The mean BCE loss over all samples.
        """
        # Ensure y_true is a column vector
        y_true = y_true.reshape(-1, 1)

        # Clip predictions to avoid log(0) and division by zero
        self.y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        self.y_true = y_true

        # Compute the mean binary cross-entropy loss
        loss = -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        return loss



    def backward(self, grad=1.0):
        """
        Perform the backward pass of the binary cross-entropy loss.

        The derivative of the mean BCE with respect to each prediction is
            dL/dp = (p - y) / (p * (1 - p)) / n
        where n is the number of samples. Since the loss is the last layer of the
        network, the incoming gradient defaults to 1.

        Parameters:
            grad (float): Gradient flowing into the loss from ahead (default 1.0).

        Returns:
            np.ndarray: Gradient of the loss with respect to y_pred,
                        of shape (n_samples, 1).
        """
        n_samples = self.y_pred.shape[0]
        # Compute the gradient of the loss with respect to predictions
        dL_dp = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred)) / n_samples
        return grad * dL_dp

    


