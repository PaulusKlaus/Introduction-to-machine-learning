import numpy as np


class LogisticRegression():

    def __init__(self, learning_rate=0.001, epochs=1000,pred_to_class=0.5):
        # NOTE: Feel free to add any hyperparameters
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        self.pred_to_class = pred_to_class

    def sigmoid_function(self, z):
        return 1/(1+np.exp(-z))
        #return 1/(1+np.e**(-sum(self.weights*x+self.bias)))

    def _compute_loss(self, y, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def compute_gradients(self, X, y, y_pred):
        N = X.shape[0]
        error = y_pred - y

        grad_w =X.T @ error
        grad_b = error  # scalar
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.bias = self.bias - self.learning_rate * (grad_b)
        self.weights = self.weights - self.learning_rate * (grad_w)

    def accuracy(self, true_values, predictions):
        # ss_res = np.sum((true_values - predictions) ** 2)
        # ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        # if ss_tot == 0:
        #     return 1.0 if ss_res == 0 else 0.0
        # return 1 - (ss_res / ss_tot)
        return np.mean(true_values == predictions)

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

        y = np.asarray(y).reshape(-1)  # (m,)
        X = np.asarray(X)  # (m, n)

        self.weights = np.zeros(X.shape[1])  # x.shape = datapunkter, features
        self.bias = 0
        # raise NotImplementedError("The fit method is not implemented yet.")
        # Gradient Descent
        for _ in range(self.epochs):
            lin_model = X @ self.weights + self.bias  # np.matmul(self.weights, X.transpose()) + self.bias   #(m, )
            y_pred = self.sigmoid_function(lin_model)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)

            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            _pred_to_class = [1 if _y > self.pred_to_class else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, _pred_to_class))
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
        y_pred = self.sigmoid_function(lin_model)

        return [1 if _y > self.pred_to_class else 0 for _y in y_pred]

        # raise NotImplementedError("The predict method is not implemented yet.")
    def predict_proba(self, X):
        lin_model = X @ self.weights + self.bias
        return self.sigmoid_function(lin_model)

