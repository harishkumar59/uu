# Logistic Regression (Iris Dataset)
#pip install pandas numpy scikit-learn matplotlib statsmodels
# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# Step 2: Load the Iris dataset
iris = datasets.load_iris()

# Step 3: Select features and target
X = iris.data[:, :2]   # Features (shape: [150, 2])
Y = iris.target        # Labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)

# Step 4: Create a logistic regression classifier
logreg = LogisticRegression(C=1e5)

# Step 5: Train the model on the dataset
logreg.fit(X, Y)

# Step 6: Define the range of the plot (a bit beyond the data range)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # Sepal length range
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # Sepal width range

# Step 7: Create a meshgrid (2D grid) for plotting decision boundaries
h = 0.02  # step size for the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Step 8: Predict the class for every point in the mesh
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Step 9: Reshape prediction results to match meshgrid shape for plotting
Z = Z.reshape(xx.shape)

# Step 10: Visualize the decision boundaries
plt.figure(1, figsize=(4, 3))

# Color the regions according to predicted classes
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Overlay the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

# Step 11: Customize plot labels and limits
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())  # Hide x-axis ticks
plt.yticks(())  # Hide y-axis ticks

# Step 12: Display the final plot
plt.show()