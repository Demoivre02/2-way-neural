# TwoHiddenLayerNeuralNetwork

This Python code implements a neural network with two hidden layers using the backpropagation algorithm for training. The network structure consists of an input layer, two hidden layers, and an output layer. The activation function used is the sigmoid function.

## Usage

1. **Initialization:**
   ```python
   test_input = numpy.array(
       [
           [0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1],
       ],
       dtype=numpy.float64,
   )
   output = numpy.array(([0], [1], [1], [0], [1], [0], [0], [1]), dtype=numpy.float64)

   neural_network = TwoHiddenLayerNeuralNetwork(input_array=test_input, output_array=output)
   ```

2. **Training:**
   ```python
   neural_network.train(output=output, iterations=10, give_loss=False)
   ```
   Train the neural network by specifying the true output values and the number of training iterations.

3. **Prediction:**
   ```python
   result = neural_network.predict(numpy.array(([1, 1, 1]), dtype=numpy.float64))
   ```
   Obtain predictions for new input values.

4. **Example:**
   ```python
   example_result = example()
   ```

## Structure

- The neural network has two hidden layers, and the number of nodes in each layer is customizable during initialization.

- The feedforward function is responsible for propagating input values through the network and producing predictions.

- The backpropagation function fine-tunes the weights of the network based on the error rate obtained in the previous iteration.

- The train function performs multiple iterations of feedforward and backpropagation to train the network.

- The predict function predicts the output for a given input using the trained neural network.

- Sigmoid and its derivative functions are used as activation functions.

## Example

The example function demonstrates how to use the neural network class. It provides fixed input and output values to the model, trains the model for a fixed number of iterations, and then makes predictions.

## References

- [Backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Sigmoid Activation Function](https://en.wikipedia.org/wiki/Sigmoid_function)
- [Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network)