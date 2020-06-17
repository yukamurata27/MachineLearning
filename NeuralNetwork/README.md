# Neural Network (MATLAB)


## Purposes
#### 1) Implement backpropagation for neural network
#### 2) Recognize hand-written digits


## Features
#### 1) Neural Network
#### 2) Backpropagation (learn weights)
#### 3) Hand-written digit recognition


## Dataset
- ex4data1.mat


## Usage
#### 1) Open MATLAB
#### 2) Open the ex4 folder in MATLAB
#### 3) In the terminal window, execute:  
```
ex4
```


## Development Flow
#### 1) Data Visualization
![Input images](img/data-visualization.jpg)
#### 2) Compute cost function J (Feedforward propagation)
```
% Format y (vector to matrix)
y_mat = zeros(m, num_labels);
for i=1:m
    y_mat(i,y(i)) = 1;
end

% Feedforward propagation

% From layer 1 (input) to layer 2 (hidden)
X = [ones(m,1) X];
a = sigmoid(X * Theta1'); % row = each example | col = a's for the example

% From layer 2 (hidden) to layer 3 (output)
a = [ones(m,1) a];
h = sigmoid(a * Theta2'); % row = each example | col = h's for the example

% Sum up all the errors between my output (h) and expected output
J = 1 / m * sum(sum(-y_mat.*log(h)-(1-y_mat).*log(1-h)));
```
#### 3) Add regularization terms
Do NOT include bias terms (1st columns)!  
```
sum_Theta1 = sum(sum(Theta1(:,2:end).^2));
sum_Theta2 = sum(sum(Theta2(:,2:end).^2));
regularization = lambda / (2*m) * (sum_Theta1 + sum_Theta2);
J = J + regularization;
```
#### 4) Compute sigmoid gradient
```
g = sigmoid(z) .* (1 - sigmoid(z));
```
#### 5) Randomize initial weights
Break symmetry.
```
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
```

## Assignment Link
- [Neural Network Learning](https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning) 
(Notice: you need to log in to see the programming assignment.)
