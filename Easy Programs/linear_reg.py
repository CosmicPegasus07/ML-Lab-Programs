import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate some example data
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Make predictions on the training data
y_pred = model.predict(x)

# Calculate and print the model's performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
print("R-squared: ", r2)
