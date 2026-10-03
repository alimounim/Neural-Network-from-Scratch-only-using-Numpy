import numpy as np
from .PLayer import PLayer

class MeanSquaredErrorLoss(PLayer):
    """
    Mean squared error (MSE) loss layer, used for regression such as predicting trip duration.
    Measures the average squared difference between predictions and targets:
        MSE(p, y) = mean( (p - y)^2 )
    Large errors are penalized much more than small ones because the error is squared.
    The network's last layer should be linear (no activation), since the target can be any real value.
    This layer has no learnable parameters.
    """

    def __init__(self):
        """
        Initialize the MeanSquaredErrorLoss layer.

        Attributes:
            y_pred (np.ndarray): Predictions from the forward pass, reused in backward.
            y_true (np.ndarray): Targets from the forward pass, reused in backward.
        """
        super().__init__(name="MeanSquaredErrorLoss")
        self.y_pred = None
        self.y_true = None



    def forward(self, y_pred, y_true):
        """
        Perform the forward pass: compute the mean squared error.

        Parameters:
            y_pred (np.ndarray): Predicted values of shape (n_samples, 1).
            y_true (np.ndarray): True values. Reshaped to match y_pred,
                                 so shape (n_samples,) or (n_samples, 1) both work.

        Returns:
            float: The mean squared error over all samples.
        """
        self.y_pred = y_pred
        self.y_true = np.asarray(y_true).reshape(-1, 1)
        return np.mean((self.y_pred - self.y_true) ** 2)



    def backward(self, grad=1.0):
        """
        Perform the backward pass of the mean squared error loss.

        The derivative of the mean squared error with respect to each prediction is
            dL/dp = 2 * (p - y) / n
        where n is the number of samples. Since the loss is the last layer of the
        network, the incoming gradient defaults to 1.

        Parameters:
            grad (float): Gradient flowing into the loss from ahead (default 1.0).

        Returns:
            np.ndarray: Gradient of the loss with respect to y_pred,
                        of shape (n_samples, 1).
        """
        n_samples = self.y_pred.shape[0]
        return grad * 2 * (self.y_pred - self.y_true) / n_samples
