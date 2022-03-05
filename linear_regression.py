"""
@author: Muhammad Hanan Asghar
"""

## LINEAR REGRESSION (Alghorithum For Linear Data)
import numpy as np

class LinearReg:
    """
    Linear Regression Class
    """
    def __init__(self, lr=0.01, epochs=800):
        self.lr = lr
        self.epochs= epochs
        self.dw = 0
        self.db = 0
        self.dz = 0
    
    def loss(self, y, y_hat):
        """
        Loss Function
        """
        # y: True Output
        # y_hat: Output From Equation

        loss = np.mean((y_hat - y) ** 2)
        return loss
    
    def fit(self, x, y):
        """
        Training Function
        """
        m, n = x.shape    
    
        # Initial Weights and Bias
        self.weights = np.zeros((n,1))
        self.bias = 0
        
        y = y.reshape(m,1)
        
        losses = []
        
        for idx in range(self.epochs):
            # Linear Regression Equation
            y_hat = np.dot(x, self.weights) + self.bias
            
            # Calculate Loss
            J = self.loss(y, y_hat)
            losses.append(J)
            
            # Calculating Gradients
            m, n = x.shape
            self.dz = y_hat - y
            self.dw = (1/m) * np.dot(x.T, self.dz)
            self.db = (1/m) * np.sum(self.dz)
            
            #Updating Gradients
            self.weights = self.weights - self.lr*self.dw
            self.bias = self.bias - self.lr*self.db
        
        return self.weights, self.bias, np.sum(losses) / len(losses)
    
    def predict(self, x):
        """
        Predict Function
        """
        return np.dot(x, self.weights) + self.bias
    
X_train, X_test, y_train, y_test = _, _, _, _
model = LinearReg(epochs=100)
w, b, l = model.fit(X_train,y_train)

preds = model.predict(x)

## THANK YOU