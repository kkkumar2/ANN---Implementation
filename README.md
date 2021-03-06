# ANN - Implementation using tensorflow

INTRODUCTION
In this article, I will explain to you the basics of neural networks and their code. Nowadays many students just learn how to code for neural networks without understanding the core concepts behind it and how it internally works. First,  Understand what is Neural Networks?

What is Neural Network?
Neural Network is a series of algorithms that are trying to mimic the human brain and find the relationship between the sets of data. It is being used in various use-cases like in regression, classification, Image Recognition and many more.

As we have talked above that neural networks tries to mimic the human brain then there might be the difference as well as the similarity between them. Let us talk in brief about it.

Some major differences between them are biological neural network does parallel processing whereas the Artificial neural network does series processing also in the former one processing is slower (in millisecond) while in the latter one processing is faster (in a nanosecond).

Architecture Of ANN
A neural network has many layers and each layer performs a specific function, and as the complexity of the model increases, the number of layers also increases that why it is known as the multi-layer perceptron.

The purest form of a neural network has three layers input layer, the hidden layer, and the output layer. The input layer picks up the input signals and transfers them to the next layer and finally, the output layer gives the final prediction and these neural networks have to be trained with some training data as well like machine learning algorithms before providing a particular problem. Now, let’s understand more about perceptron.

About Perceptron
As discussed above multi-layered perceptron these are basically the hidden or the dense layers. They are made up of many neurons and neurons are the primary unit that works together to form perceptron. In simple words, as you can see in the above picture each circle represents neurons and a vertical combination of neurons represents perceptrons which is basically a dense layer.

About Perceptron ANN

Now in the above picture, you can see each neuron’s detailed view. Here, each neurons have some weights (in above picture w1, w2, w3) and biases and based on this computations are done as, combination = bias + weights * input (F = w1*x1 + w2*x2 + w3*x3) and finally activation function is applied output = activation(combination) in above picture activation is sigmoid represented by      1/(1 + e-F). There are some other activation functions as well like ReLU, Leaky ReLU, tanh, and many more.

Working Of ANN
At First, information is feed into the input layer which then transfers it to the hidden layers, and interconnection between these two layers assign weights to each input randomly at the initial point. and then bias is added to each input neuron and after this, the weighted sum which is a combination of weights and bias is passed through the activation function. Activation Function has the responsibility of which node to fire for feature extraction and finally output is calculated. This whole process is known as Foreward Propagation. After getting the output model to compare it with the original output and the error is known and finally, weights are updated in backward propagation to reduce the error and this process continues for a certain number of epochs (iteration). Finally, model weights get updated and prediction is done.


# How to use this Module

## The Coding is done in a way that u don't have to build the code, u just need to change the data in the configuration file(yaml)

### A glimpse of what is present in the configuration file is mentioned below

```yaml

params:
  epochs: 5
  batch_size: 32
  no_classes: 10
  input_shape: [28,28]
  loss_function: sparse_categorical_crossentropy
  metrics: accuracy
  optimizer: SGD
  validation_datasize: 5000



artifacts:
  artifacts_dir: artifacts
  model_dir: model
  plots_dir: plots
  checkoint_dir: checkpoints
  model_name: model.h5
  plots_name: plot.png

logs:
  logs_dir: logs_dir
  general_logs: general_logs
  tensorboard_logs: tensorboard_logs

```

## A glimpse of the Layers

```python

    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputlayer"),
          tf.keras.layers.Dense(300,activation="relu", name="hiddenlayer1"),
          tf.keras.layers.Dense(100,activation="relu", name="hiddenlayer2"),
          tf.keras.layers.Dense(OUTPUT_CLASSES,activation="softmax", name="outputlayer")] 

```
```python
tensorboard --logdir logs_dir/tensorboard_logs
```