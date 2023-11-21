# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
data = {
    'feature1': 2 * np.random.rand(100),
    'feature2': 3 * np.random.rand(100),
    'feature3': 4 * np.random.rand(100),
    'target': 5 + 2 * np.random.randn(100)
}
df = pd.DataFrame(data)

# Split the data into features and target
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multivariable linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

# Plot the regression line
fig = plt.figure(figsize=(12, 6))

# 3D Scatter plot for the training data
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['feature1'], X_train['feature2'], y_train, c='blue', marker='o', label='Actual values')

# Plot the regression plane
x1_range = np.linspace(X['feature1'].min(), X['feature1'].max(), 100)
x2_range = np.linspace(X['feature2'].min(), X['feature2'].max(), 100)
x1, x2 = np.meshgrid(x1_range, x2_range)
y_pred_plane = model.intercept_ + model.coef_[0] * x1 + model.coef_[1] * x2
ax.plot_surface(x1, x2, y_pred_plane, alpha=0.5, color='red', label='Regression plane')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Regression Plane and Training Data')

plt.show()
