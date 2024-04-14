import numpy as np
import os
import pandas as pd

rng = np.random.default_rng()

class Adaline(object):
    
    def __init__(self):
        #  Having it on random does change the learning process
        self.eta = 0.01 
        #self.weights = np.zeros(units) Boundary wont change bcz the weights are all 0
        self.bias = np.float64(0.)
        self.losses = []

    def fit(self, x, y, epochs):
        self.weights = rng.normal(loc=0.0, scale=0.01, size=x.shape[1])

        for _ in range(epochs):
            lossVal = 0 # will hold the amount of errors in each epoch
            update = 0
            
            for xi1, target1 in zip(x,y):
              update += (-2*(target1 - (np.dot(self.weights,xi1)+self.bias)))/len(y)
      
              loss = (update/-1)**2
              update = -self.eta*update

              self.weights += update*xi1 # The rate of change is directly propotionate to the xi
              self.bias += update

              # This will update the error to the loss, until the final loop of xi will set the final loss value - To lazy to do it efficiently 
              lossVal = loss
            self.losses.append(lossVal) 
                
        return self
    
    def net_input(self, x):
        return np.dot(x, self.weights) + self.bias
    
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