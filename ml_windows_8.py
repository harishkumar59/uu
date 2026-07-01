# 1) IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 2) LOAD DATA
test  = pd.read_csv("california_housing_test.csv")
train = pd.read_csv("california_housing_train.csv")

print("--- Train Head ---")
print(train.head())
print("\n--- Train Tail ---")
print(train.tail())

print("\n--- Train Info ---")
print(train.info())
print("\n--- Test Info ---")
print(test.info())

# 3) SAVE TARGET & COMBINE FOR PREPROCESSING
n_train = train.shape[0]
n_test  = test.shape[0]
y       = train['median_house_value'].values

data = pd.concat((train, test)).reset_index(drop=True)
data.drop(['longitude', 'latitude'], axis=1, inplace=True)

# 4) OPTIONAL: QUICK VISUALIZATIONS
plt.figure()
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.show()

sns.lmplot(x='median_income',      y='median_house_value', data=train)
plt.show()

sns.lmplot(x='housing_median_age', y='median_house_value', data=train)
plt.show()

sns.pairplot(train, palette='rainbow')
plt.show()

# 5) FEATURE SELECTION & MISSING-VALUE IMPUTATION
data = data[['total_rooms', 'total_bedrooms',
             'housing_median_age', 'median_income',
             'population', 'households']]
print("\n--- Selected Data Info ---")
data.info()

for col in data.columns:
    data.fillna(data.mean(), inplace=True)

# Split back into train / test
train = data[:n_train]
test  = data[n_train:]

# 6) TRAIN/VALIDATION SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2, random_state=42
)

# 7) SCALE FEATURES & TARGET
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)    # Use transform, not fit_transform

sc_y    = StandardScaler()
# Scale target values as column-vectors, then flatten back to 1D
y_train = sc_y.fit_transform(y_train.reshape(-1,1)).ravel()
y_test  = sc_y.transform(   y_test.reshape(-1,1)).ravel()

# 8) FIT THE SVR MODEL
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

# 9) PREDICT & INVERSE-SCALE THE OUTPUT
y_pred = regressor.predict(X_test)  
# Bring predictions back to the original scale
y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1)).flatten()

# 10) COMPARE REAL VS. PREDICTED
df = pd.DataFrame({
    'Real Values'     : sc_y.inverse_transform(y_test.reshape(-1,1)).flatten(),
    'Predicted Values': y_pred
})
print("\n--- Predictions Evaluation ---")
print(df.head())