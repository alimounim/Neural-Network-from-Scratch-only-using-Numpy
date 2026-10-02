import numpy as np
from PLayer import PLayer

class SigmoidFunction(PLayer):
    """
    Sigmoid activation function layer. Applies the sigmoid function element-wise to the input.
    The sigmoid function is defined as: sigmoid(x) = 1 / (1 + exp(-x))
    """

    def __init__(self):
        """
        Initialize the SigmoidFunction layer.
        """
        super().__init__(name="SigmoidFunction")
        self.out = None # To store the output of the forward pass for use in the backward pass
    

    def forward(self, x):
        """
        Perform the forward pass of the sigmoid activation function.

        Parameters:
            x (np.ndarray): Input data of any shape.

        Returns:
            np.ndarray: Output data after applying the sigmoid function, same shape as input.
        """
        self.out = 1 / (1 + np.exp(-x))  # Apply sigmoid function
        return self.out  # Return the output of the sigmoid function

    def backward(self, grad):
        """
        Perform the backward pass of the sigmoid activation function.

        Parameters:
            grad (np.ndarray): Gradient of the loss with respect to the output of this layer,
                               same shape as output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
                        same shape as input.
        """
        sigmoid_derivative = self.out * (1 - self.out) # Derivative of sigmoid function
        return grad * sigmoid_derivative  # Chain rule: multiply incoming gradient by the derivative of the sigmoid 