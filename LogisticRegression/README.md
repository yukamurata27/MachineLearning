# Logistic Regression (MATLAB)


## Purpose
#### 1) Predicts university admission
#### 2) Predicts QA test result of microchips from a fabrication plant


## Dependency
- Optimization Toolbox


## Features
#### 1) Logistic Regression
#### 2) Regularized Logistic Regression


## Dataset
#### 1) ex2data1.txt (logistic regression)
#### 2) ex2data2.txt (regularized logistic regression)


## Usage
#### 1) Open MATLAB
#### 2) Open the ex2 folder in MATLAB
#### 3) In the terminal window, execute the function name you like to run.  
Logistic regression:
```
ex2
```
Regularized logistic regression:
```
ex2_reg
```


## Development Flow (Logistic Regression)
#### 1) Data Visualization
![Scatter plot](img/data-plot.jpg)
#### 2) Implement the Sigmoid function
```
g = 1 ./ (1 + exp(-z));
```
#### 3) Compute the cost function and gradient
```
h = sigmoid(X * theta);
J = 1 / m * (- y' * log(h) - (1-y)' * log(1-h));
grad = 1 / m * X' * (h - y);
```
#### 4) Find optimal learning parameters using fminunc
See the result section below for the decision boundary.
```
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```
#### 5) Make predictions
Don't forget to use the sigmoid function!
```
p = round(sigmoid(X * theta));
```
#### 6) Result
Accuracy = 89.0%  
![Decision boundary plot](img/decision-boundary.jpg)

## Development Flow (Regularized Logistic Regressione)
#### 1) Data Visualization
![Scatter plot](img/data-plot2.jpg)
#### 2) Feature Mapping
- Linear decision boundary does not fit the dataset, so create more features.
- Map the features into all polynomial terms of *x1* and *x2* up to the sixth power.
#### 3) Compute the cost function and gradient with regularization
```
h = sigmoid(X * theta);
J = 1 / m * (-y' * log(h) - (1-y)' * log(1-h)) + lambda / (2*m) * sum(theta(2:end).^2);
grad = 1 / m * X' * (h - y) + [0; lambda / m * theta(2:end)];
```
#### 4) Find optimal learning parameters using fminunc
Similar to Step 4 in **Development Flow (Logistic Regression)** above.  
See the result section below for the decision boundary.
#### 5) Make predictions
#### 6) Tweak lambda (regularization parameter)
Good lambda value is 1
#### 7) Results
Accuracy = 83.1%  
Good prediction (lambda = 1):  
![Decision boundary plot](img/decision-boundary2.jpg)
Overfitting (lambda = 0):  
![Decision boundary plot (overfitting)](img/overfitting.jpg)
Underfitting (lambda = 100):  
![Decision boundary plot (underfitting)](img/underfitting.jpg)


## Assignment Link
- [Logistic Regression](https://www.coursera.org/learn/machine-learning/programming/ixFof/logistic-regression) 
(Notice: you need to log in to see the programming assignment.)
