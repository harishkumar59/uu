# Train an SVM regressor on the California Housing Dataset
#pip install pandas numpy matplotlib seaborn scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2) LOAD DATA
test = pd.read_csv("california_housing_test.csv")
train = pd.read_csv("california_housing_train.csv")

train.head()
train.tail()

print(train.info())
print(test.info())

# 3) SAVE TARGET & COMBINE FOR PREPROCESSING
n_train = train.shape[0]
n_test = test.shape[0]
y = train["median_house_value"].values

data = pd.concat((train, test)).reset_index(drop=True)
data.drop(["longitude", "latitude"], axis=1, inplace=True)

# 4) OPTIONAL: QUICK VISUALIZATIONS
plt.figure()
sns.heatmap(data.corr(), cmap="coolwarm")
plt.show()

sns.lmplot(x="median_income", y="median_house_value", data=train)
sns.lmplot(x="housing_median_age", y="median_house_value", data=train)
sns.pairplot(train, palette="rainbow")

# 5) FEATURE SELECTION & MISSING-VALUE IMPUTATION
data = data[
    [
        "total_rooms",
        "total_bedrooms",
        "housing_median_age",
        "median_income",
        "population",
        "households",
    ]
]
data.info()

# Fixed: Filled missing values across the dataframe once
data.fillna(data.mean(), inplace=True)

# split back into train / test
train = data[:n_train]
test = data[n_train:]

# 6) TRAIN/VALIDATION SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2, random_state=42
)

# 7) SCALE FEATURES & TARGET
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # <-- use transform, not fit_transform

sc_y = StandardScaler()
# keep y_train/y_test 1D for sklearn, but scale them as column-vectors then ravel
y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = sc_y.transform(y_test.reshape(-1, 1)).ravel()

# 8) FIT THE SVR MODEL
from sklearn.svm import SVR

regressor = SVR(kernel="rbf")
regressor.fit(X_train, y_train)  # no more DataConversionWarning

# 9) PREDICT & INVERSE-SCALE THE OUTPUT
y_pred = regressor.predict(X_test)
# bring back to original scale
y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# 10) COMPARE REAL VS. PREDICTED
df = pd.DataFrame(
    {
        "Real Values": sc_y.inverse_transform(y_test.reshape(-1, 1)).flatten(),
        "Predicted Values": y_pred,
    }
)
print(df.head())