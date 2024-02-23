import numpy as np
from sklearn.model_selection import train_test_split

class LogRegression:
    def __init__(self, learning_rate = 0.01, max_iterations = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.likelihoods = []
        
        self.epsilon = 1e-7 # Since log(0) is not defined, create a minimum value epsilon to replace with if necessary

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def log_likelihood(self, y_star, y_hat):
        # Avoid undefined log(0) behavior
        # Courtesy of Implementation of Gradient Ascent using Logistic Regression article on Medium
        y_hat = np.maximum(np.full(y_hat.shape, self.epsilon), np.minimum(np.full(y_hat.shape, 1 - self.epsilon), y_hat))
        
        # Remember that the MLE is y * log(yh) + (1-y) * log(1-yh)
        likelihood = (y_star * np.log(y_hat) + (1 - y_star) * np.log(1 - y_hat))
    
        return np.mean(likelihood)
    
    def gradient_ascent(self, X, y, display_weights=True):
        self.weights = np.zeros((X.shape[1]))
        
        # Run Gradient Ascent
        for i in range(self.max_iterations):
            y_hat = self.sigmoid(np.dot(X, self.weights))
            
            gradient = np.mean((y - y_hat) * X.T, axis=1)
            
            self.weights += self.learning_rate * gradient

            # Show weights at different iterations to see how fast or slow convergence is
            if i % 100 == 0 and display_weights:
                print(f"Weights at {i} iterations: ")
                print(self.weights)

            likelihood = self.log_likelihood(y, y_hat)

            self.likelihoods.append(likelihood)
    
    def pred_prob(self,X):
        if self.weights is None:
            raise Exception("Fit the model before predicting")
      
        probabilities = self.sigmoid(np.dot(X,self.weights))
        
        return probabilities
    
    def predict(self, X, threshold = 0.5):
        bin_hats = np.array(list(map(lambda x: 1 if x >threshold else 0, self.pred_prob(X))))
        
        return bin_hats

data = np.loadtxt("titanic_data.csv", delimiter=',', skiprows=1)

y = data[:, 0]
X = data[:, 1:-1]

min_val = np.min(X)
max_val = np.max(X)

X_scaled = (X - min_val) / (max_val - min_val)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.15, random_state=0) 
       
model = LogRegression(max_iterations=2000)
model.gradient_ascent(x_train, y_train)

y_hat = model.predict(x_test)

print('Final Weights: ')
print(model.weights)
print('MLEs of theta hat: ')
print(model.likelihoods)