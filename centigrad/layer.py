from variable import Variable

class Layer:
    def __init__(self, 
                 input_dim, 
                 n_neurons, 
                 activation = "sigmoid",
                initial_weights=None,
                initial_biases=None):
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.activation = activation
        # Initial Weights
        if initial_weights is not None:
            self.weights = initial_weights
        else:
            self.weights = np.random.normal(loc=0, scale=1, size=(input_dim,n_neurons))
        self.weights = np.vectorize(Variable)(self.weights)
        # Initial Biases
        if initial_biases is not None:
            self.biases = initial_biases
        else:
            self.biases = np.random.normal(loc=0, scale=1, size = (n_neurons, 1))
        self.biases = np.vectorize(Variable)(self.biases)
    def __repr__(self):
        return f"Layer(neurons={self.n_neurons}, activation = {self.activation})"
    def forward(self, input_data):
        self.input_dim = input_data.shape[0]
        self.output_dim = input_data.shape[1]
        linear_combination = np.matmul(self.weights.T, input_data) + self.biases
        iteration_indexes = np.ndindex(linear_combination.shape[0], 
                                        linear_combination.shape[1])
        for entry in iteration_indexes:
            activation = getattr(linear_combination[entry], self.activation)
            linear_combination[entry] = activation()
        return linear_combination
    def backward(self):
        gradient_method = getattr(Variable, "grad")
        return np.vectorize(gradient_method)(self.weights), np.vectorize(gradient_method)(self.biases)

