# https://github.com/NikMehraDev/One-Neuron
import numpy as np
import matplotlib.pyplot as plt

class Format_report:
    def __init__(self, X, pred, Y=None) -> None:
        if Y is not None:
            for x, y, p, approx_p in zip(X, Y, pred, pred.round(3)):
                print(f'\033[1;48;5;247m{x}\033[0m: Expected = \033[1;48;5;240m{y}\033[0m, Prediction = {p} ~ \033[1;48;5;249m{approx_p}\033[0m.')
        else:
            for x, p, approx_p in zip(X, pred, pred.round(3)):
                print(f'\033[48;5;247m{x}\033[0m: Prediction = \033[48;5;240m{p}\033[0m ~ \033[48;5;250m{pred.round()}\033[0m.')
        print()
        
def analyze_predictions(inputs, actual_output, o1, o2):
    """
    Analyzes and visualizes the changes in predictions made by an ANN.
    
    Parameters:
    - inputs: List or array of inputs (x-axis for the graph)
    - actual_output: List or array of actual target values
    - o1: List or array of initial predictions made by the model
    - o2: List or array of final predictions made by the model
    
    Outputs:
    - A graph comparing the actual output with initial and final predictions.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot actual outputs
    plt.plot(inputs, actual_output, label="Actual Output", color="green", linewidth=2)
    
    # Plot initial predictions
    plt.plot(inputs, o1, label="Initial Predictions (o1)", linestyle="--", color="red", linewidth=2)
    
    # Plot final predictions
    plt.plot(inputs, o2, label="Final Predictions (o2)", linestyle="-.", color="blue", linewidth=2)
    
    # Enhancing the visualization
    plt.title("ANN Prediction Analysis", fontsize=16, fontweight='bold')
    plt.xlabel("Inputs", fontsize=12)
    plt.ylabel("Outputs", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Display the graph
    plt.tight_layout()
    plt.show()

np.random.seed(291)

def f(x):
    """A Linear Function"""
    return x*1 +2

def generate_dataset(fnx= f, no_data_set=None):
    _x = np.random.randn(np.random.randint(2,11) or no_data_set , 1)
    _y = fnx(_x)
    return (_x, _y)

X, Y = generate_dataset(no_data_set=100)
class Neuron:
    def __init__(self) -> None:
        self.weight = np.random.randn(1 , 1)
        self.bias = np.zeros((1, 1))

    def predict(self, inputs):
        self.output = inputs.dot(self.weight.T) + self.bias
        return self.output
        
    def backward(self, x, y, learning_rate=1e-2) -> None:
        y_cap = self.predict(inputs=x)
        # Compute gradients
        dw = -2 * np.mean((y - y_cap) * x)
        db = -2 * np.mean(y - y_cap)
        
        # Update weight and bias
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        
    def train(self, x, y, epochs=1000, learning_rate=1e-2) -> None:
        for epoch in range(epochs):
            self.backward(x, y, learning_rate=learning_rate)
            progress = (epoch + 1) / epochs
            bar_length = 20  # Length of the progress bar
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '>' + ' ' * (bar_length - filled_length)
            percentage = round(progress * 100, 2)
            
            print(f"[Training {bar} {percentage:.1f}%]", end='\r')

        print()
            # Forward pass: Predict outputs
            
    @property
    def loss(self, x=X, y=Y):
        return np.mean((y - self.predict(x)) ** 2)
    
    def ask_me(self) -> None:
        user_inputs = []
        i = 1
        while i <= self.n_inputs:
            try:
                user_inputs.append(int(input(f"Value({i}): ")))
                i += 1
            except:
                print("Invalid input. Please enter an integer.")
                    
        output = self.predict(np.array([user_inputs]))
        Format_report(X=user_inputs, pred=output)

n = Neuron()

o1 = n.predict(X)
n.train(x=X, y=Y, epochs=5000, learning_rate=1e-1)
o2 = n.predict(X)
print(f'Weight: {n.weight} --- Bias: {n.bias}')
print(f'Loss {n.loss}')

analyze_predictions(X, Y, o1, o2)
