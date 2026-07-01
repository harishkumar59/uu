# Implement Linear Regression (Diabetes Dataset)
#pip install pandas numpy scikit-learn matplotlib statsmodels
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split

# Step 1: Load the built-in diabetes dataset from scikit-learn
diabetes = datasets.load_diabetes()

# Step 2: Select only one feature (BMI - body mass index at index 2)
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Step 3: Split the data into training and testing sets (70% train, 30% test)
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(
    diabetes_X, diabetes.target, test_size=0.3
)

# Step 4: Create a LinearRegression model object
regr = linear_model.LinearRegression()

# Step 5: Train the model using the training data (fit finds the best-fit line)
regr.fit(diabetes_X_train, diabetes_y_train)

# Step 6: Make predictions on the test set using the trained model
diabetes_y_pred = regr.predict(diabetes_X_test)

# Step 7: Print the slope (coefficient) of the regression line
print('Coefficients: \n', regr.coef_)

# Step 8: Print the Mean Squared Error (how far off our predictions are, on average)
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Step 9: Print the R² score (how well the model explains the variance; 1 = perfect)
print('Coefficient of determination (R² score): %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Step 10: Visualize the results
# Plot the actual test data points (black dots)
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')

# Plot the regression line (predicted values) in blue
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# Hide x and y ticks for a cleaner look
plt.xticks(())
plt.yticks(())

# Add axis labels
plt.xlabel('Blood Sugar Level')
plt.ylabel('Disease Progression')

# Show the final plot
plt.show()