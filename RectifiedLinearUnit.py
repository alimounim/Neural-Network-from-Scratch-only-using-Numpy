import numpy as np
from PLayer import PLayer

class RectifiedLinearUnit(PLayer):
    """
    Rectified Linear Unit (ReLU) activation function layer.
    Applies ReLU element-wise to the input: relu(x) = max(0, x).
    Positive values pass through unchanged; negative values become 0.
    This layer has no learnable parameters.
    """

    def __init__(self):
        """
        Initialize the RectifiedLinearUnit layer.

        Attributes:
            mask (np.ndarray): Boolean array marking where the input was
                               positive. Set during forward and reused in
                               backward.
        """
  
        super().__init__(name="RectifiedLinearUnit")
        self.mask = None  # To store the mask of positive inputs during forward pass


    def forward(self, x):
        """
        Perform the forward pass of the ReLU activation function.

        Parameters:
            x (np.ndarray): Input data of any shape.

        Returns:
            np.ndarray: max(0, x) applied element-wise, same shape as input.
        """
        self.mask = x > 0  # Create a mask of where the input is positive
        return np.maximum(0, x)  # Apply ReLU element-wise


    def backward(self, grad):
        """
        Perform the backward pass of the ReLU activation function.

        The derivative of ReLU is 1 where the input was positive and 0
        elsewhere, so the incoming gradient passes through only where
        the input was positive.

        Parameters:
            grad (np.ndarray): Gradient of the loss with respect to the output of this layer,
                               same shape as output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
                        same shape as input.
        """
        return grad * self.mask  # Pass gradient only where input was positive
    

