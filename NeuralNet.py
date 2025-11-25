import numpy as np


class NeuralNet:
    def __init__(self, layers, epochs=1000, learning_rate=0.001, momentum=0.9, function='relu', validation_split=0.2):
        self.L = len(layers) #layers
        self.n = layers.copy() #Array with number of neurons in each layer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.function = function # activation function
        self.validation_split = validation_split   
    
        self.h = [
            np.zeros(layers[lay]) for lay in range(self.L)
        ] #Array of arrays for the fields
          
        self.xi = [
            np.zeros(layers[lay]) for lay in range(self.L)
        ] #Array of arrays for the activations
        
        # He initialization (correct for ReLU)
        self.w = [None] + [
            np.random.randn(layers[lay], layers[lay - 1]) * np.sqrt(2. / layers[lay - 1])
            for lay in range(1, self.L)
        ] #an array of matrices for the weights
        
        # Small biases (don't destroy activations)
        self.theta = [
            np.zeros(layers[lay]) for lay in range(self.L)
        ] #Array of arrays for the thresholds
        
        self.delta = [
            np.zeros(layers[lay]) for lay in range(self.L)
        ] #Array of arrays for the propagation errors
        
        self.d_w = [
            np.zeros((1, 1)) if lay == 0
            else np.zeros((layers[lay], layers[lay - 1]))
            for lay in range(self.L)
        ] #Array of matrices for the changes of weights
        
        self.d_theta = [
             np.zeros(layers[lay]) for lay in range(self.L)
        ] # an array of arrays for the changes of the weights
        
        self.d_w_prev = [
            np.zeros((1, 1)) if lay == 0
            else np.zeros((layers[lay], layers[lay - 1]))
            for lay in range(self.L)
        ] #an array of matrices for the previous changes of the weights, used for the momentum term
        
        self.d_theta_prev = [
             np.zeros(layers[lay]) for lay in range(self.L)
        ] # an array of arrays for the previous changes of the thresholds, used for the momentum term
        
        self.training_error = []
        self.validation_error = []

        self.get_activation_function = {
            "sigmoid": self.sigmoid,
            "relu": self.relu,
            "linear": self.linear,
            "tanh": self.tanh
        }

        self.get_derivative = {
            "sigmoid": self.sigmoid_derivative,
            "relu": self.relu_derivative,
            "linear": self.linear_derivative,
            "tanh": self.tanh_derivative
        }

    # computation of activation function 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def linear(self, x):
        return x
    
    def tanh(self, x):
        return np.tanh(x)

    #  computation of derivative of the activation function
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def linear_derivative(self, x):
        return np.ones_like(x)
    
    def tanh_derivative(self, x):
        t = np.tanh(x)
        return 1 - t**2

            
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.fact = self.get_activation_function.get(self.function)
        self.fact_derivative = self.get_derivative.get(self.function)

        n_samples = X.shape[0]
        n_val = int(n_samples * self.validation_split)
    
        indices = np.random.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
            
        for epoch in range(self.epochs):
            # loop through all patterns in random order
            for i in np.random.permutation(X_train.shape[0]):
                x_patron = X_train[i].reshape(-1)
                y_patron = y_train[i].reshape(-1)
    
                self.feed_foward(x_patron)
                self.backpropagation(y_patron)
                self.update_weights_and_thresholds()

            train_error = self.mean_squared_error(X_train, y_train)
            val_error = self.mean_squared_error(X_val, y_val)

            self.training_error.append(train_error)
            self.validation_error.append(val_error)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs} - Training Error: {train_error:.6f} - Validation Error: {val_error:.6f}")


    def feed_foward(self, x: np.ndarray):
        self.xi[0] = x
        for lay in range(1, self.L):
            self.h[lay] = np.dot(self.w[lay], self.xi[lay - 1]) - self.theta[lay]
            self.xi[lay] = self.fact(self.h[lay])

    def backpropagation(self, y: np.ndarray):
        self.delta[-1] = self.fact_derivative(self.h[-1]) * (self.xi[-1] - y)
        for lay in range(self.L - 2, 0, -1):
            self.delta[lay] = self.fact_derivative(self.h[lay]) * np.dot(self.w[lay + 1].T, self.delta[lay + 1])

    def update_weights_and_thresholds(self):
        for lay in range(1, self.L):
            self.d_w[lay] = -self.learning_rate * np.outer(self.delta[lay], self.xi[lay - 1]) + self.momentum * self.d_w_prev[lay]
            self.d_w_prev[lay] = self.d_w[lay]
            self.w[lay] += self.d_w[lay]
        
            self.d_theta[lay] = self.learning_rate * self.delta[lay] + self.momentum * self.d_theta_prev[lay]
            self.d_theta_prev[lay] = self.d_theta[lay]
            self.theta[lay] += self.d_theta[lay]

    def predict(self, x: np.ndarray):
        predictions = []
        for i in range(x.shape[0]):
            self.feed_foward(x[i]) #Feed forward for each input sample
            predictions.append(self.xi[-1]) #Output of layer activation
        return np.array(predictions)
    
    def mean_squared_error(self, X: np.ndarray, y: np.ndarray):
        error = 0.0
        for i in range(X.shape[0]):
            self.feed_foward(X[i]) 
            y_pred = self.xi[-1]   
            error += np.sum((y_pred - y[i]) ** 2)
        total_error = error / X.shape[0]
        return total_error
        
    def loss_epochs(self):
         return np.array(self.training_error), np.array(self.validation_error)