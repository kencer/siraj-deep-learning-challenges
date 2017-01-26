# Challenge: How to Make a Neural Network - Intro to Deep Learning #2
# This code uses a computational graph approach for backpropagation

from numpy import exp, array, random, dot

class Layer():
    def __init__(self, num_inputs, num_outputs):

        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single layer, with num_inputs and num_outputs connections.
        # We assign random weights to a matrix, with values in the range -1 to 1
        # and mean 0.
        self.weights = 2 * random.random((num_inputs, num_outputs)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    # Derivative of the error function w.r.t the weights in this layer
    def __derivative_weights(self, inputs, next_layer_derivative):
        return dot(inputs.T, next_layer_derivative * self.__sigmoid_derivative(dot(inputs, self.weights)))

    # Derivative of the error function w.r.t. to the inputs from the previous layer. This derivative then backpropagates
    # to the layer before it
    def derivative_inputs(self, inputs, next_layer_derivative):
        return dot(next_layer_derivative * self.__sigmoid_derivative(dot(inputs, self.weights)), self.weights.T)

    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.weights))

    # Adjusts weights using gradient descent
    def adjust_weights(self, alpha, inputs, next_layer_derivative):
        self.weights += alpha * -1 * self.__derivative_weights(inputs, next_layer_derivative)


class NeuralNetwork():

    def __init__(self):

        # Initializes the layers

        self.hidden_layer_1 = Layer(3, 3)
        self.hidden_layer_2 = Layer(3, 3)
        self.output_layer = Layer(3, 1)

    # We train the neural network through a process of trial and error.
    # Adjusting the weights for the layers each time
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate=0.1):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error
            # Note that output comes first since dE/dy_hat = d/dy_hat 1/2 * (y_actual - y_hat)^2
            # = (y_actual - y_hat) * (-1) = y_hat - y_actual
            error = output - training_set_outputs

            # Backpropagate the gradients

            # Adjust weights for the output layer
            self.output_layer.adjust_weights(learning_rate, self.hidden_layer_2_outputs, error)
            output_layer_derivative_inputs = self.output_layer.derivative_inputs(self.hidden_layer_2_outputs, error)

            # Adjust weights for the hidden layer 2
            self.hidden_layer_2.adjust_weights(learning_rate, self.hidden_layer_1_outputs, output_layer_derivative_inputs)
            hidden_layer_2_derivative_inputs = self.hidden_layer_2.derivative_inputs(self.hidden_layer_1_outputs, output_layer_derivative_inputs)

            # Adjust weights for the hidden layer 1
            self.hidden_layer_1.adjust_weights(learning_rate, training_set_inputs, hidden_layer_2_derivative_inputs)

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network's successive layers and stores the ouputs
        # The ouputs will be used later for backpropagation
        self.hidden_layer_1_outputs = self.hidden_layer_1.think(inputs)
        self.hidden_layer_2_outputs = self.hidden_layer_2.think(self.hidden_layer_1_outputs)
        self.output_layer_outputs = self.output_layer.think(self.hidden_layer_2_outputs)
        return self.output_layer_outputs


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print "Hidden Layer 1:" , neural_network.hidden_layer_1.weights
    print "Hidden Layer 2:" , neural_network.hidden_layer_2.weights
    print "Output Layer:" , neural_network.output_layer.weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    learning_rate = 0.5

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000, learning_rate)

    print "New synaptic weights after training: "

    print "Hidden Layer 1:" , neural_network.hidden_layer_1.weights
    print "Hidden Layer 2:" , neural_network.hidden_layer_2.weights
    print "Output Layer:" , neural_network.output_layer.weights

    # Test the neural network with existing situations.
    print "Considering existing situation [0, 0, 1] -> ?: "
    print neural_network.think(array([0, 0, 1]))

    print "Considering existing situation [1, 0, 1] -> ?: "
    print neural_network.think(array([1, 0, 1]))

    print "Considering existing situation [0, 1, 1] -> ?: "
    print neural_network.think(array([0, 1, 1]))

    print "Considering existing situation [1, 1, 1] -> ?: "
    print neural_network.think(array([1, 1, 1]))

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))
