import random
import numpy
#import mnist


class Neuron:

    def __init__(self, layer_number, node_number, bias):
        self.layer_number = layer_number
        self.node_number = node_number
        self.bias = bias
        self.activation = 0
        self.error = 0
        self.mini_batch_error_sum = 0
        self.mini_batch_error_activation_sum = 0
        self.z = 0
        # neurons are stored in an array
        # the index of the neurons in the previous layer is used as the key for this dictionary
        # the value of the dict entries is a pair of activation value and weight from the previous layer
        # this could maybe be streamlined
        self.inputs = {}

    # adds a neuron to the dictionary of inputs
    def establish_input_neuron(self, neuron_id, weight):
        # defaults input value to 0
        # index 2 is used to sum the product of error in current node and activation value of previous node
        # to aid calculation when updating weights with gradient descent
        self.inputs[neuron_id] = [0, weight, 0]

    # output inputs dictionary keys so the outputs of these neurons can be obtained from outside this object
    def get_input_nodes(self):
        return list(self.inputs.keys())

    # with the neuron ids outputted from the function above, their output values are obtained and pushed back
    # into this object with this function.
    def set_input_activation(self, neuron_id, activation):
        self.inputs[neuron_id][0] = activation

    # once all inputs are collected they can be processed
    def process_inputs(self):
        # sigmoid function
        summation = self.bias
        for input_node in self.inputs:
            summation += self.inputs[input_node][0] * self.inputs[input_node][1]
        # set z for later use
        self.z = summation
        # 1 / (1 + e^-z)
        self.activation = 1 / (1 + numpy.exp(-summation))

    def get_activation(self):
        return self.activation


# adds set_value method specifically for neurons in the first layer
class InputNeuron(Neuron):

    def set_activation(self, activation):
        self.activation = activation


class NeuralNet:

    def __init__(self, data):
        # create network structure

        self.dataset_size = len(data)
        self.data = data
        self.learning_rate = 1.0
        # default values
        # first layer is an input layer so number of neurons equals dataset size
        self.layers = (self.dataset_size, 4, self.dataset_size)
        self.default_bias = 1
        self.default_weighting = 1

        # express neural_net as 2D list
        self.network = [[] for layer in self.layers]

        self.output_vector = []

        # iterate through each layer
        for layer_id in range(0, len(self.layers)):
            # iterate through each node in layer
            for node_id in range(0, self.layers[layer_id]):
                # if input layer, create InputNeurons that can have their value manually set
                if layer_id == 0:
                    node = InputNeuron(layer_id, node_id, numpy.random.rand())
                # otherwise set number of inputs equal to number of nodes in previous layer
                else:
                    node = Neuron(layer_id, node_id, numpy.random.rand())

                    # connect each node to every node in the previous layer
                    for next_layer_node_id in range(0, self.layers[layer_id-1]):
                        # default weight to 1
                        node.establish_input_neuron(next_layer_node_id, numpy.random.rand())

                # add node to neural net list
                self.network[layer_id].append(node)

    def pass_data(self, sample):
        # pass data through network

        # feed data to first layer
        for i in range(0, len(sample)):
            # set values for input layer
            self.network[0][i].set_activation(sample[i])

        # iterate through each layer after input layer
        for layer_id in range(1, len(self.network)):
            # iterate through each neuron in layer
            for neuron in self.network[layer_id]:
                # get data from previous layer
                input_nodes = neuron.get_input_nodes()
                for input_neuron in input_nodes:
                    # get the outputted value from node in previous layer
                    input_neuron_activation = self.network[layer_id-1][input_neuron].get_activation()
                    # set the input value for this node
                    neuron.set_input_activation(input_neuron, input_neuron_activation)

                # apply sigmoid to inputted data
                neuron.process_inputs()

                # if in output layer, add value to output_vector
                if layer_id == len(self.network)-1:
                    self.output_vector.append(neuron.get_activation())

    def gradient_descent(self, num_epochs, mini_batch_size):

        # for each epoch
        for i in range(0, num_epochs):

            # randomly divide dataset into equally-sized mini batches
            mini_batches = []

            for mini_batch in mini_batches:
                for sample in mini_batch:

                    # fix this to take input
                    self.pass_data(sample)

                    # backpropagate
                    self.backprop(sample[0], sample[1])

                    # iterate through forwards, ignoring input layer
                    for layer_num in range(1, len(self.network)):
                        for neuron in self.network[layer_num]:
                            # use the current network state to add to the error/activation sums
                            # used in calculating the changes in weight/bias after mini batch is wholly fed through
                            neuron.mini_batch_error_sum += neuron.error
                            for input in neuron.inputs:
                                # set the error/activation product sum value as the product of this neuron's error
                                # and the activation of the neuron associated with this weight, from the prev layer
                                input[2] += neuron.error * self.network[layer_num-1][input[0]].activation

                # once mini batch has been fed through network
                # iterate through every neuron in every layer
                for layer in self.network:

                    for neuron in layer:

                        # calculate quantity to change bias by:
                        # the product of the mean of its errors and the learning rate
                        bias_delta = neuron.mini_batch_error_sum * self.learning_rate / mini_batch_size
                        neuron.bias -= bias_delta

                        # adjust each weight
                        for input in neuron.inputs:
                            # calculate mean of products of error in secondary neuron and activation in primary neuron
                            # multiply by learning rate
                            weight_delta = input[2] * self.learning_rate / mini_batch_size
                            input[1] -= weight_delta


    def backprop(self, x, y):
        # recall Z is the sum of weight/activation products + bias

        # pass input x through network

        # calculate error in output layer
        for neuron in self.layers[-1]:
            neuron.error = (neuron.activation - y) * self.sigmoid_prime(neuron.z)

        # propagate backwards through each layer the precedes the output layer:
        for l in range(2, len(self.layers)):
            # for each neuron in layer -l
            for neuron in self.layers[-l]:
                sum = 0
                # for each neuron in next layer
                for successive_neuron in self.layers[1-l]:
                    # multiplies the next-layer neuron's error
                    # by the weight connecting it to the neuron in the CURRENT layer
                    # weights between two nodes are stored in the node that appears later in the network
                    # (e.g. weight between a pair of neurons in layers 1 and 2 is stored in the node in layer 2)
                    sum += successive_neuron.error * successive_neuron.inputs[neuron.node_number][1]
                # update the current neuron's error
                neuron.error = sum * self.sigmoid_prime(neuron.z)


    def sigmoid_prime(self, z):
        return 1.0 / (1.0 + numpy.exp(-z))

    def output_result(self):
        for x in self.output_vector:
            print(x)


number_samples = 1000
number_input_neurons = 8

# generate a list of samples and outputted data
samples = []
for i in range(0, number_samples):
    sample = []
    for j in range(0, number_input_neurons):
        sample.append(random.uniform(0, 1))
    outputs = []
    for y in sample:
        # for now, the expected outputs are half the inputs
        outputs.append(y/2)
    samples.append([sample, outputs])


# pass each dataset through the neural net and store the outputs
for sample in samples:
    test_net = NeuralNet(sample[0])
    test_net.pass_data()
    output = test_net.output_vector
    sample.append(output)


# use the cost function to evaluate outputs

# cost function
summation = 0
for sample in samples:
    # find vector difference between expected and real outcome
    vector_difference = []
    # for each output value
    for i in range(0, len(sample[1])):
        # find difference between expected and actual output value, put this in a difference vector
        vector_difference.append(sample[1][i] - sample[2][i])

    # find square magnitude by doing pythagoras and square rooting, then square again
    # last two steps cancel out so just find sum of squares
    square_sum = 0

    for x in vector_difference:
        square_sum += x**2

    summation += square_sum

cost = summation / (2 * number_samples)

print(cost)
