# centigrad

This project began as a blind replication of Andrej Karpathy's micrograd repository(https://github.com/karpathy/micrograd/tree/master). I call it a "blind replication" because I initially started by watching Andrej's YouTube video, [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), from his "Neural Networks: Zero to Hero series".

Inspired by Andrej's approach to building things from scratch, I watched the video for a while, grasping key concepts and details, before pausing it to begin my own replication, intending to later compare my solution with his. After reaching a stable state in my implementation, I returned to Andrej's video. To my surprise, I found notable similarities between our approaches, which I will discuss further in this document. I may consider undertaking similar replications for the rest of his series.

At the time of this replication, I had little to no experience with TensorFlow or PyTorch. After reaching a stable state in my implementation, I returned to Andrej's video to complete it. I may consider undertaking similar replications for the rest of his series.

# Installation

```
pip install centigrad
```

# Run tests

```
pytest
```

# Usage

Below are examples demonstrating the usage of the three classes in the package, including key supported operations and essential methods. For more detailed information about the implementation, please refer to the corresponding notebooks named after each object.

## Variable

```
from centigrad.variable import Variable

a = Variable(5)
b = Variable(10)
fab = (a + b)**2 + a*b + b**3
print("Parents:\n")
print(f"a's parents: {a.parents}") # Must print a list with 2 parents: Variable(15) (a+b), and Variable(50) (a*b)
print(f"a's first parent parents: {a.parents[0].parents}") # Must print 1 parent: Variable(value=225) (a + b)**2
print()
print(f"b's parents: {b.parents}") # Must print a list with 3 parents: Variable(15) (a+b), Variable(50) (a*b)
                                    # and Variable(1000) (b**3)
print(f"b's third parent parents: {b.parents[2].parents}") # Must print 1 parent: Variable(value=1275) (a + b)**2
                                                           # which comes from the total expression fab

print("Gradients:\n")
print(f"Gradient of fab wrt a: {a.grad():.2f}") # Must print 40
print(f"Gradient of fab wrt b: {b.grad():.2f}") # Must print 335

```

## Layer

```
initial_weights = np.array([[1, 0],
                            [0, 1]])
initial_biases = np.array([[1],
                           [0]])
layer = Layer(input_dim = 2,
              n_neurons = 2,
              activation = "relu")
```

## Network

```
X = np.array([[0, 0, 1, 1],
       [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

nn = Network(input_dim = 2,
             layers_widths = [2, 1], 
             layers_activations = ["relu", "linear"])
nn.fit(X, y, epochs = 1000, learning_rate = 0.01)

print("Initial loss:", "{:.20f}".format(nn.epoch_losses[0]))
print(f"Final loss after {len(nn.epoch_losses)-1} epocs:", "{:.20f}".format(nn.epoch_losses[-1]))
```

# License

MIT