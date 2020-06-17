function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 %

% Format y (vector to matrix)
y_mat = zeros(m, num_labels);
for i=1:m
    y_mat(i,y(i)) = 1;
end

% Feedforward propagation

% From layer 1 (input) to layer 2 (hidden)
X1 = [ones(m,1) X];
a = sigmoid(X1 * Theta1'); % row = each example | col = a's for the example

% From layer 2 (hidden) to layer 3 (output)
a = [ones(m,1) a];
h = sigmoid(a * Theta2'); % row = each example | col = h's for the example

% No need to cleanup h

% Sum up all the errors between my output (h) and expected output
J = 1 / m * sum(sum(-y_mat.*log(h)-(1-y_mat).*log(1-h)));

% Add regularization terms (do NOT include bias terms)
sum_Theta1 = sum(sum(Theta1(:,2:end).^2));
sum_Theta2 = sum(sum(Theta2(:,2:end).^2));
regularization = lambda / (2*m) * (sum_Theta1 + sum_Theta2);
J = J + regularization;

% Part 2 %

D1 = zeros(hidden_layer_size,1);
D2 = zeros(num_labels,1);

% For each example
for t = 1:m
    % A) Feedforward propagation

    a1 = [1; X(t,:)'];
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);

    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % B) Backpropagation

    % Get errors in layer 3
    d3 = a3 - y_mat(t,:)';
    
    % Get errors in layer 2
    mult = Theta2' * d3;
    d2 = mult(2:end) .* sigmoidGradient(z2);
    
    % C) Accumulate the gradient
    D1 = D1 + d2 * a1';
    D2 = D2 + d3 * a2';
end

regularization = lambda / m * [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta1_grad = D1 / m + regularization;

regularization = lambda / m * [zeros(num_labels,1) Theta2(:,2:end)];
Theta2_grad = D2 / m + regularization;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
