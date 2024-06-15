## Perceptron Training Rule
## Overview
The Perceptron is one of the simplest types of artificial neural networks used for binary classification tasks. This project demonstrates how to implement the Perceptron training rule, train a Perceptron on a dataset, and calculate the accuracy of the model.

## Features
- Perceptron Training: Implementation of the Perceptron training algorithm.
- Accuracy Calculation: Evaluate the model's performance by calculating the accuracy on a test dataset.
- Binary Classification: Applicable to binary classification problems.
- Customizable Parameters: Learning rate and number of iterations can be adjusted.
## Contents
- Perceptron Training Rule
- Dataset
- Usage
- Example
- Contributing
## Perceptron Training Rule
The Perceptron training rule updates the weights based on the prediction error for each training sample. The rule is as follows:
- Initialize weights (including bias) to small random numbers.
- For each training sample:
- Compute the output using the current weights.
- Update the weights based on the error:
  wi<-wi+
𝑤𝑖←𝑤𝑖+𝜂(𝑦−𝑦^)𝑥𝑖w i←wi+η(y− y^)xi
​
where 
𝑤𝑖= is the weight for feature.
η is the learning rate, 
y is the true label, 
y^ is the predicted label, and 
𝑥𝑖 is the feature value.


![graphtr](https://github.com/Ifra-Zaib/Machine-Learning-Perceptron-training-rule/assets/172352661/5e398c2a-78fa-49c2-b7e8-15941ec9ded7)

![tr](https://github.com/Ifra-Zaib/Machine-Learning-Perceptron-training-rule/assets/172352661/16c3e273-b805-4dee-890c-ed4b812b1fb0)
