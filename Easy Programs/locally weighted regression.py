import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import lsq_linear

def locally_weighted_regression(x, y, query_x, tau):
    """
    Perform locally weighted regression on the given data.
    
    x : array-like
        The independent variable.
    y : array-like
        The dependent variable.
    query_x : array-like
        The independent variable for which to predict the dependent variable.
    tau : float
        The width of the Gaussian kernel.
    """
    # Add a column of ones to x for the bias term
    x = np.column_stack((x, np.ones(len(x))))
    
    # Initialize the weight matrix
    weights = np.zeros((len(x), len(x)))
    
    # Fill the weight matrix with the Gaussian kernel
    for i in range(len(x)):
        for j in range(len(x)):
            weights[i, j] = np.exp(-euclidean(x[i], x[j])**2 / (2 * tau**2))
    
    # Initialize the prediction array
    predictions = np.zeros(len(query_x))
    
    # Perform linear regression for each query point
    for i, query in enumerate(query_x):
        # Add a column of ones to the query point for the bias term
        query = np.append(query, 1)
        
        # Compute the weights for the query point
        query_weights = np.exp(-euclidean(x, query)**2 / (2 * tau**2))
        
        # Compute the locally weighted linear regression coefficients
        beta = lsq_linear(x.T @ np.diag(query_weights) @ x, x.T @ np.diag(query_weights) @ y)
        
        # Compute the predicted value for the query point
        predictions[i] = query @ beta.x
    
    return predictions
