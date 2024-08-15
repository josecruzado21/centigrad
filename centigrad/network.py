from .layer import Layer

class Network:
    def __init__(self, 
                 input_dim, 
                 dimension, 
                 activations, 
                 cost_function = "l2", 
                 epochs=10,
                 initial_weights = None,
                 initial_biases = None):
        if (len(dimension)>0) & (len(dimension)==len(activations)):
            self.depth = len(dimension)
            self.dimension = dimension
            self.activations = activations
            self.cost_function = cost_function
            self.layers = []
            for idx, val in enumerate(dimension):
                if len(self.layers)==0:
                    input_dim_layer = input_dim
                else: input_dim_layer = dimension[idx-1]
                self.layers.append(Layer(input_dim = input_dim_layer,
                                        n_neurons = val, 
                                         activation = activations[idx],
                                        initial_weights = initial_weights[idx] if initial_weights is not None else None,
                                        initial_biases = initial_biases[idx] if initial_biases is not None else None))
        else:
            raise("The network should be at least 1 deep long")
    
    
    def cost(self,X, y):
        prediction = self.forward(X)
        if self.cost_function == "l2":
            return (((prediction - y)**2).sum().value)/(y.shape[1])
                      
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data     