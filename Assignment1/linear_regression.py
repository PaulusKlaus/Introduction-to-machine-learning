import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, epochs =10000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate= learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []


    def _function(self, x):
        return x

    def _compute_loss(self, y, y_pred):
        return 0.5 * np.mean((y - y_pred) ** 2)


    def compute_gradients(self, X, y, y_pred):
        N = X.shape[0]
        error = y_pred - y

        grad_w = (X.T @ error) / N            # shape: (n,)
        grad_b = np.mean(error)               # scalar
        return grad_w, grad_b


    def update_parameters(self, grad_w, grad_b):
        self.bias = self.bias - self.learning_rate*(grad_b)
        self.weights = self.weights - self.learning_rate*(grad_w)


    def accuracy(self,true_values, predictions):
        # Better: R² score for regression
        ss_res = np.sum((true_values - predictions)**2)
        ss_tot = np.sum((true_values - np.mean(true_values))**2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_tot)

        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement

        # Ensure the correct shape 

        y = np.asarray(y).reshape(-1)         # (m,)
        X = np.asarray(X)                     # (m, n)

        self.weights = np.zeros(X.shape[1])  # x.shape = datapunkter, features
        self.bias = 0
        #raise NotImplementedError("The fit method is not implemented yet.")
        # Gradient Descent
        for _ in range(self.epochs):

            lin_model = X @ self.weights + self.bias#np.matmul(self.weights, X.transpose()) + self.bias   #(m, )
            y_pred = self._function(lin_model)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)

            self.update_parameters(grad_w, grad_b)
            
            loss = self._compute_loss(y, y_pred)
            #pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, y_pred))
            self.losses.append(loss)

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        lin_model = X @ self.weights + self.bias
        y_pred = self._function(lin_model)
        return y_pred

        #raise NotImplementedError("The predict method is not implemented yet.")

