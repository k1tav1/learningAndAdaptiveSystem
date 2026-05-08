import numpy as np

# 1. Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Dataset (XOR Problem)
# Inputs: [A, B] | Output: A XOR B
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
expected_output = np.array([[0], [1], [1], [0]])

# 3. Initialization
np.random.seed(42) # For consistent results
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# 4. Training Loop
epochs = 10000
learning_rate = 0.1

print("--- Training Started ---")
for i in range(epochs):
    # Forward Prop
    hidden_layer_activation = sigmoid(np.dot(inputs, hidden_weights))
    predicted_output = sigmoid(np.dot(hidden_layer_activation, output_weights))
    
    # Calculate Error
    error = expected_output - predicted_output
    
    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
    
    # Updating Weights
    output_weights += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate

    # Print error every 2000 steps to show progress
    if i % 2000 == 0:
        print(f"Epoch {i} Error: {np.mean(np.abs(error)):.4f}")

# 5. Final Output
print("\n--- Final Predictions After Learning ---")
print(predicted_output)