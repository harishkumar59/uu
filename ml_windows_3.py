# Logistic Regression with Iris Dataset
#pip install pandas numpy scikit-learn matplotlib statsmodels
# Step 1: Import Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import statsmodels.api as sm  # For detailed statistical summary
from sklearn import datasets

# Step 2: Load the Iris Dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# Step 3: Split the Data (80% training, 20% testing)
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and Train the Logistic Regression Model
# Note: multi_class='multinomial' is removed as it is deprecated/removed in newer scikit-learn versions
log_reg = LogisticRegression(solver='newton-cg', max_iter=200)
log_reg.fit(trainX, trainY)

# Step 5: Make Predictions
y_pred = log_reg.predict(testX)

# Step 6: Evaluate the Model
print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
print('Error rate: {:.2f}'.format(1 - accuracy_score(testY, y_pred)))

# Step 7: Cross-Validation Scores
clf = LogisticRegression(solver='newton-cg', max_iter=200)
scores = cross_val_score(clf, trainX, trainY, cv=5)
print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(testY, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualize the Confusion Matrix
plt.matshow(conf_matrix, cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Step 9: Predicted Class Probabilities
probability = log_reg.predict_proba(testX)
print("\nPredicted Probabilities:\n", probability)

# Convert to DataFrame for Better Visualization
df = pd.DataFrame(probability, columns=log_reg.classes_)

# Step 10: Verify Probabilities Sum to 1
df['sum'] = df.sum(axis=1)

# Step 11: Add Predicted and Actual Classes
df['predicted_class'] = y_pred
df['actual_class'] = testY.reset_index(drop=True)

# Step 12: Check if Predictions are Correct
df['correct_prediction?'] = df['predicted_class'] == df['actual_class']

# Step 13: Manually Calculate Accuracy
true_predictions = df['correct_prediction?'].sum()
total = df.shape[0]
print('Manual Calculated Accuracy is: {:.2f}%'.format((true_predictions / total) * 100))

# Step 14: Inspect Misclassified Instances
wrong_pred = df[df["correct_prediction?"] == False]
print("\nMisclassified Predictions:\n", wrong_pred)

# Step 15: Multinomial Logit Model with Statsmodels
X_sm = sm.add_constant(X)
mnlogit_mod = sm.MNLogit(y, X_sm)
mnlogit_fit = mnlogit_mod.fit(maxiter=100)
print(mnlogit_fit.summary())