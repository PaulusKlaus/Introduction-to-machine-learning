import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.1, epochs =1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate= learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []


    def _function(self, x):
        return x

    def _compute_loss(self, y, y_pred):
        return 1/(2*len(y))*sum((y-y_pred)**2)


    def compute_gradients(self, x, y, y_pred):
        _grad_b0 = 1/len(y) * np.sum(y_pred-y)
        _grad_b1 = 1/len(y) * np.sum(x*(y_pred-y))
        return _grad_b1, _grad_b0


    def update_parameters(self, grad_b0, grad_b1):
        self.bias = self.bias - self.learning_rate*(grad_b0)
        self.weights = self.weights - self.learning_rate*(grad_b1)


    def accuracy(true_values, predictions):
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

        self.weights = np.zeros(X.shape[1])  # x.shape = datapunkter, features
        self.bias = 0
        #raise NotImplementedError("The fit method is not implemented yet.")
        # Gradient Descent
        for _ in range(self.epochs):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias
            y_pred = self._function(lin_model)
            grad_b1, grad_b0 = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_b1, grad_b0)
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
        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self._function(lin_model)
        return y_pred

        #raise NotImplementedError("The predict method is not implemented yet.")





# import numpy as np
#
# class LinearRegression:
#     def __init__(self, fit_intercept=True, method="normal", l2=0.0,
#                  lr=1e-2, n_iter=1000, tol=1e-8, random_state=None):
#         """
#         Args:
#             fit_intercept (bool): whether to learn an intercept term.
#             method (str): "normal" (closed-form) or "gd" (gradient descent).
#             l2 (float): L2 regularization strength (ridge). 0.0 disables it.
#             lr (float): learning rate (used when method='gd').
#             n_iter (int): max iterations (used when method='gd').
#             tol (float): early-stop tolerance on parameter change (gd).
#             random_state (int|None): seed for reproducibility (gd init).
#         """
#         self.fit_intercept = fit_intercept
#         self.method = method
#         self.l2 = float(l2)
#         self.lr = float(lr)
#         self.n_iter = int(n_iter)
#         self.tol = float(tol)
#         self.random_state = random_state
#
#         self.coef_ = None       # shape (n_features,)
#         self.intercept_ = 0.0
#         self._fitted = False
#
#     @staticmethod
#     def _add_bias(X):
#         return np.c_[np.ones((X.shape[0], 1)), X]
#
#     def fit(self, X, y):
#         """
#         Estimates parameters for the regressor
#
#         Args:
#             X (array<m,n>): features
#             y (array<m>): targets
#         """
#         X = np.asarray(X, dtype=float)
#         y = np.asarray(y, dtype=float).reshape(-1)
#
#         if X.ndim != 2:
#             raise ValueError("X must be a 2D array (m, n).")
#         if y.ndim != 1:
#             raise ValueError("y must be a 1D array (m,).")
#         if X.shape[0] != y.shape[0]:
#             raise ValueError("X and y must have the same number of rows.")
#
#         m, n = X.shape
#
#         if self.method == "normal":
#             # Build design matrix with/without bias
#             if self.fit_intercept:
#                 Xd = self._add_bias(X)  # (m, n+1)
#             else:
#                 Xd = X  # (m, n)
#
#             # Ridge-regularized normal equation:
#             # theta = (X^T X + λI)^(-1) X^T y
#             # Do not regularize the bias term.
#             XtX = Xd.T @ Xd
#             if self.l2 > 0.0:
#                 I = np.eye(XtX.shape[0])
#                 if self.fit_intercept:
#                     I[0, 0] = 0.0  # don't penalize intercept
#                 XtX = XtX + self.l2 * I
#
#             Xty = Xd.T @ y
#             theta = np.linalg.pinv(XtX) @ Xty  # stable pseudo-inverse
#
#             if self.fit_intercept:
#                 self.intercept_ = float(theta[0])
#                 self.coef_ = theta[1:].astype(float)
#             else:
#                 self.intercept_ = 0.0
#                 self.coef_ = theta.astype(float)
#
#         elif self.method == "gd":
#             rng = np.random.default_rng(self.random_state)
#             # Initialize parameters
#             if self.fit_intercept:
#                 w = rng.normal(scale=1e-3, size=n + 1)  # [b, w1..wn]
#                 Xd = self._add_bias(X)
#             else:
#                 w = rng.normal(scale=1e-3, size=n)      # [w1..wn]
#                 Xd = X
#
#             lam = self.l2
#             prev_w = w.copy()
#             for _ in range(self.n_iter):
#                 # predictions and gradient
#                 y_pred = Xd @ w
#                 residuals = y_pred - y  # shape (m,)
#
#                 grad = (Xd.T @ residuals) / m  # shape like w
#
#                 if lam > 0.0:
#                     # L2 on weights only (no bias)
#                     if self.fit_intercept:
#                         reg = np.r_[0.0, lam * w[1:]]
#                     else:
#                         reg = lam * w
#                     grad = grad + reg
#
#                 # parameter update
#                 w = w - self.lr * grad
#
#                 # early stopping on parameter movement
#                 if np.linalg.norm(w - prev_w) < self.tol:
#                     break
#                 prev_w = w.copy()
#
#             if self.fit_intercept:
#                 self.intercept_ = float(w[0])
#                 self.coef_ = w[1:].astype(float)
#             else:
#                 self.intercept_ = 0.0
#                 self.coef_ = w.astype(float)
#
#         else:
#             raise ValueError("method must be 'normal' or 'gd'.")
#
#         self._fitted = True
#         return self
#
#     def predict(self, X):
#         """
#         Generates predictions
#
#         Args:
#             X (array<m,n>): features
#
#         Returns:
#             array<m>: predictions
#         """
#         if not self._fitted:
#             raise RuntimeError("Call .fit(X, y) before .predict(X).")
#
#         X = np.asarray(X, dtype=float)
#         if X.ndim != 2:
#             raise ValueError("X must be a 2D array (m, n).")
#         if X.shape[1] != self.coef_.shape[0]:
#             raise ValueError(
#                 f"X has {X.shape[1]} features but model was fit with {self.coef_.shape[0]}."
#             )
#
#         return X @ self.coef_ + self.intercept_
