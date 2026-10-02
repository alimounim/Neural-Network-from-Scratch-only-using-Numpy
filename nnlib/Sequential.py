import pickle
from .PLayer import PLayer

class Sequential(PLayer):
    """
    A container that chains layers together into a single network.
    The Sequential class holds an ordered list of layers. The forward pass feeds the input through
    each layer in order, and the backward pass sends the gradient through the layers in reverse order.
    Because it inherits from PLayer, a whole network behaves like a single layer, so a Sequential
    can even be placed inside another Sequential. It also updates, saves and loads the parameters
    of all its layers. The loss function is kept outside the Sequential and called separately.
    """

    def __init__(self, layers=None):
        """
        Initialize the Sequential container.

        Parameters:
            layers (list[PLayer], optional): Layers to start the network with, in order.
                                             If None, the network starts empty.

        Attributes:
            layers (list[PLayer]): The ordered list of layers that make up the network.
        """
        super().__init__(name = "Sequential")
        self.layers = list(layers) if layers is not None else []





    def add(self, layer):
        """
        Append a layer to the end of the network.

        Parameters:
            layer (PLayer): The layer to add. It becomes the last layer of the network.
        """
        self.layers.append(layer)




    def forward(self, x):
        """
        Perform the forward pass through every layer in order.
        The output of each layer becomes the input of the next one.

        Parameters:
            x (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Output of the last layer.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x



    def backward(self, grad):
        """
        Perform the backward pass through every layer in reverse order.
        Each layer computes the gradients of its own parameters and returns the gradient
        with respect to its input, which is passed to the layer behind it.

        Parameters:
            grad (np.ndarray): Gradient of the loss with respect to the output of the network,
                               usually the result of loss.backward().

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the network.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad



    def params(self):
        """
        Collect the parameters and gradients of every layer in the network.
        Calls params() on each layer and combines the results, so layers without parameters
        contribute nothing and nested Sequential layers are handled automatically.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: All (parameter, gradient) pairs, in layer order.
        """
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.params())
        return all_params 



    def step(self, lr):
        """
        Update every parameter in the network using gradient descent:
            parameter = parameter - lr * gradient
        The update is done in place so that each layer's own arrays are changed.
        Should be called after backward().

        Parameters:
            lr (float): Learning rate, the size of each update step.
        """
        for param, grad in self.params():
            param -= lr * grad  # Update in place

        




    def save(self, path):
        """
        Save all model weights to a file.
        Only the parameter values are saved (not the architecture), so the same
        network must be built again before loading them.

        Parameters:
            path (str): File path to save the weights to, e.g. "models/XOR_solved.w".
        """
        weights = []
        for param, grad in self.params():
            weights.append(param.copy())  # Save a copy of the parameter values
        with open(path, 'wb') as f:
            pickle.dump(weights, f)  # Save only the parameter values



    def load(self, path):
        """
        Load model weights from a file into this network.
        The network must have the same architecture as the one that was saved.
        The weights are copied into the existing arrays in place.

        Parameters:
            path (str): File path to load the weights from.

        Raises:
            ValueError: If the number or shapes of the saved weights do not match
                        the parameters of this network.
        """
        with open(path, 'rb') as f:
            weights = pickle.load(f)  # Load the saved parameter values
        
        params = self.params() # Get the current parameters of the network
        expected = [param.shape for param, _ in params]
        found = [w.shape for w in weights]

        if expected != found:
            raise ValueError(f"Saved weights {found} do not match the network's parameters {expected}.")

        for (param, _), saved in zip(params, weights):
            param[...] = saved  # Copy the saved values into the existing parameter arrays


