""" 
David Fernando Armendárirz Torres | A01570813
TC3006C.102 | Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)
 """

""" 
=====================================
===PREPARE PROGRAMMING ENVIRONMENT===
=====================================
"""
# Import libraries
import pandas as pd # Dataframe
import numpy as np # Math
import seaborn as sns # Graphs
import matplotlib.pyplot as plt # Graphs
import random # Randomness
import numpy as np # Math
from sklearn import svm # Support Vector Machines
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, validation_curve # Model tunning
from sklearn.preprocessing import StandardScaler # Preprocessing
from sklearn.metrics import accuracy_score, classification_report # Model Evaluation

# Ensure reproducibility
sd = 666
random.seed(sd)
np.random.seed(sd)



""" 
======================
===DATA EXPLORATION===
======================
"""
# Load dataset: Iris
columns = ["sepal length","sepal width","petal length","petal width", "class"]
df = pd.read_csv('iris.data', names = columns)
print(df.head(), "\n\n")

# Explore dataset
print(df.describe(), "\n\n")

# Check for null values
print(df.info(), "\n\n")

# Check target balance
print(df["class"].value_counts(), "\n\n")

# Check for linear separablity
sns.pairplot(df, hue='class', diag_kind='auto', height=1.5)
plt.title("CLOSE THE PLOT TO CONTINUE CODE EXECUTION", y = 0.9, x = -1, fontsize = 16, color = "red")

# Display the plot
plt.show()

# Check for linear separability (validation)
'''
A linear svm model is trained with semi hard marigin on all the data and predictions 
are made on the same data used to train it. If the accuracy is 100% then the 
data was properly separated otherwise the data is not linearly separable
'''
XLin = df[["sepal length", "sepal width", "petal length", "petal width"]]
yLin = df["class"]
linModel = svm.SVC(C=1e10, kernel='linear')
linModel.fit(XLin, yLin)
y_pred_lin = linModel.predict(XLin)
accuracy_lin = accuracy_score(yLin, y_pred_lin)
print("Linear Separability Accuracy: ", accuracy_lin * 100)

'''
**Important Remarks:**
- The data has no null values
- The classes are balanced
- The data is not linearly separable
'''



""" 
========================
===DATA PREPROCESSING===
========================
"""
# Split the dataset into features and targets
y = np.array(df["class"].copy().to_numpy())
X = np.array(df.drop("class", axis = 1).copy())

# Standarize the feature dataset | {Comment if undesired}
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the datasets into training and testing | 70% - 30%  | {Comment if Nested Cross Validation is employed}
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = sd, shuffle = True, stratify = y)



""" 
================
===MODELLING ===
================
"""
# Define cross validation grid
cv_grid = {"C": [0.1, 1, 10, 100], "kernel": ["linear", "poly", "rbf", "sigmoid"]}

# Define the cross validation strategy | 5-Folds 
cv = KFold(n_splits = 5, shuffle = True, random_state = sd) # Inner CV
outer_cv = KFold(n_splits = 5, shuffle = True, random_state = sd) # Outer CV

# Initialize the SVM model
model = svm.SVC(break_ties = True, random_state = sd)

# Perform cross validation grid search 
best_model = GridSearchCV(estimator = model, param_grid = cv_grid, cv = cv, scoring = "accuracy", refit = True) # Inner Loop
best_model.fit(X, y)

print("Grid Search Evaluation: ", best_model.cv_results_)

# Select the best model
print("Best parameters: ", best_model.best_params_)



""" 
======================
===MODEL EVALUATION===
======================
"""
# Evaluate the model | Accuracy | Nested Cross validation
nested_score = cross_val_score(best_model, X = X, y = y, cv = outer_cv, scoring = "accuracy") # Outer Loop
accuracy = nested_score.mean()

print("Nested Accuracies: ", nested_score)
print("Mean Accuracy: ", accuracy)

# Validation Curve on kernel parameter
train_scores, test_scores = validation_curve(
    svm.SVC(C = 10, random_state=sd),
    X, y,
    param_name = "kernel",
    param_range=cv_grid["kernel"],
    cv = cv,
    scoring="accuracy"
)

# Calculate mean and standard deviations
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.title("Validation Curve with SVM")
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.1)

plt.plot(cv_grid["kernel"], train_scores_mean, label = "Training score", color = "darkorange", lw = 2)
plt.fill_between(cv_grid["kernel"], train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.2, color = "darkorange", lw = 2)

plt.plot(cv_grid["kernel"], test_scores_mean, label = "Cross-validation score", color = "navy", lw = 2)
plt.fill_between(cv_grid["kernel"], test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.2, color = "navy", lw = 2)

plt.legend(loc = "best")
plt.grid(True)
plt.show()

# Validation Curve on C parameter
train_scores, test_scores = validation_curve(
    svm.SVC(kernel='linear', random_state=sd),
    X, y,
    param_name = "C",
    param_range = cv_grid["C"],
    cv = cv,
    scoring="accuracy"
)

# Calculate mean and standard deviations
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize = (10, 6))
plt.title("Validation Curve with SVM")
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
#plt.ylim(0.9, 1.02)

plt.semilogx(cv_grid["C"], train_scores_mean, label = "Training score", color = "darkorange", lw = 2)
plt.fill_between(cv_grid["C"], train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.2, color = "darkorange", lw = 2)

plt.semilogx(cv_grid["C"], test_scores_mean, label = "Cross-validation score", color = "navy", lw = 2)
plt.fill_between(cv_grid["C"], test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.2, color = "navy", lw = 2)

plt.legend(loc = "best")
plt.grid(True)
plt.show()

# Validation Curve on Folds
train_scores = []
test_scores = []

for train_index, test_index in outer_cv.split(X):
    X_cv, X_test_cv = X[train_index], X[test_index]
    y_cv, y_test_cv = y[train_index], y[test_index]
    
    best_model.fit(X_cv, y_cv)
    train_scores.append(best_model.score(X_cv, y_cv))
    test_scores.append(best_model.score(X_test_cv, y_test_cv))

# Calculate mean and standard deviations
train_scores_mean = np.mean(train_scores)
train_scores_std = np.std(train_scores)
test_scores_mean = np.mean(test_scores)
test_scores_std = np.std(test_scores)

# Display the values
print(f"Mean Training Score: {train_scores_mean:.4f}")
print(f"Training Score Standard Deviation: {train_scores_std:.4f}")
print(f"Mean Cross-Validation Score: {test_scores_mean:.4f}")
print(f"Cross-Validation Score Standard Deviation: {test_scores_std:.4f}")

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.title("SVM with C=10 and Linear Kernel")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.82, 1.02)

plt.plot(range(1, len(train_scores) + 1), train_scores, 'o-', label='Training score', color='darkorange')
plt.fill_between(range(1, len(train_scores) + 1), train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='darkorange')
plt.axhline(y=train_scores_mean, color='darkorange', linestyle='--', label='Mean Training score')

plt.plot(range(1, len(test_scores) + 1), test_scores, 'o-', label='Cross-validation score', color='navy')
plt.fill_between(range(1, len(test_scores) + 1), test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='navy')
plt.axhline(y=test_scores_mean, color='navy', linestyle='--', label='Mean Cross-validation score')

plt.legend(loc="best")
plt.grid(True)
plt.show()