from mnist_loader import load_data_wrapper
from neural_network import Network


training_data, validation_data, test_data = load_data_wrapper()


# Here is where we decide the number and size of each layer
# For our test, the input layer MUST have 784 neurons and the output MUST have 10
# Otherwise, you can add whatever layers you like

# Examples:
# [784, 100, 10]
# [784, 30, 30, 10]
# [784, 40, 30, 20, 10]
# [784, 10]
net = Network([784, 30, 10])

net.SGD(training_data,
        epochs = 30,            # Number of times we train on the entire dataset
        mini_batch_size = 10,   # Number of examples used for each update
        eta = 3.0,              # Learning rate
        test_data=test_data)
