import numpy as np
import os
import pandas as pd

rng = np.random.default_rng()

# Adaline - Adaptive Linear Neuron Classifier
# Net input function -> Activation function -> Update Weights
# loss function = 1/nΣ(y-y^)^2 | MSE in this case
# ∂l/∂w = -2/nΣ(y-y^)*xi
# ∂l/∂b = -2/nΣ(y-y^)
# δw = -η * ∂L/∂w
# δb = -η * ∂L/∂b
 

class Adaline(object):
    
    def __init__(self, x, y):
        #  Having it on random does change the learning process
        self.eta = 0.0001 
        #self.weights = np.zeros(units) Boundary wont change bcz the weights are all 0
        self.bias = np.float64(0.)
        self.X = x
        self.y = y
        self.losses = []
        self.weights = rng.normal(loc=0.0, scale=0.01, size=x.shape[1])
        
    def fit(self, epochs):

        for _ in range(epochs):
            change_w = -self.eta * self.calculate_gradient_descent_w(self.X, self.y)
            change_b = -self.eta * self.calculate_gradient_descent_b(self.X, self.y)
            
            self.weights = self.weights + change_w
            self.b = self.bias + change_b
            loss = self.calculate_loss(self.X, self.y)
            self.losses.append(loss)

        return self
    
    def net_input(self, xi):
        return np.dot(xi, self.weights) + self.bias
    
    def calculate_loss(self, x, y):
        loss = 0
        for xi, target in zip(x,y):
            z = self.net_input(xi)
            loss_i = (target - z)**2
            loss += loss_i
            
        return loss/x.shape[0]
        
    def calculate_gradient_descent_w(self, x, y):
        loss = 0
        for xi, target in zip(x,y):
            z = self.net_input(xi)
            for xji in xi:
                loss_i = (target - z) * xji
                loss += loss_i
        gradient_descent = -2 * (loss/len(x))
        return gradient_descent

            
    def calculate_gradient_descent_b(self, x, y):
        loss = 0
        for xi, target in zip(x,y):
            z = self.net_input(xi)
            loss_i = (target - z)
            loss += loss_i
        gradient_descent = -2 * (loss/len(x)) 
        return gradient_descent
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)
                          
    def test(self, x, y):
        correct_predictions = 0
        total_predictions = len(x)
        
        for xi, target in zip(x, y):
            z = np.dot(self.weights, xi) + self.bias
            
            if z >= 0.0:
                prediction = 1
            else:
                prediction = 0
            
            if prediction == target:
                correct_predictions += 1
            else:
                print(f'Incorrect Prediction: {prediction}, Target: {target}, Input: {xi}')
        
        accuracy = correct_predictions / total_predictions
        return accuracy