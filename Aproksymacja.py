import numpy as np
import matplotlib.pyplot as plt

# funkcja ackleya
def ackley(x1, x2):
    return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))) -
            np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.exp(1) + 20)

# generowanie losowych punktow
def generate_data(num_samples=1000):
    x1 = np.random.uniform(-2, 2, num_samples)
    x2 = np.random.uniform(-2, 2, num_samples)
    y = ackley(x1, x2)

    y_min, y_max = 0, 20
    y = 2 * (y - y_min) / (y_max - y_min) - 1

    return np.vstack((x1, x2)), y.reshape(1, -1)

# inicjalizacja wag i biasow
def initialize_weights(input_size, hidden_size, output_size):
    w_hidden = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
    w_output = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
    bias_hidden = np.zeros((hidden_size, 1))
    bias_output = np.zeros((output_size, 1))
    return w_hidden, w_output, bias_hidden, bias_output

#funkcje aktywacji
#relu
def relu(z):
    return np.maximum(0, z)
def relu_derivative(z):
    return (z>0).astype(float)

#leaky relu
def leaky_relu(z):
    return np.maximum(0.01 * z, z)
def leaky_relu_derivative(z):
    return np.where(z > 0, 1, 0.01)

#sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    return z * (1 - z)

# elu
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))
def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1, alpha * np.exp(z))

# swish
def swish(z):
    return z * sigmoid(z)
def swish_derivative(z):
    sig = sigmoid(z)
    return sig + z * sig * (1 - sig)

#tanh
def tanh(z):
    return np.tanh(z)

activation_functions = {
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'swish': (swish, swish_derivative),
    'elu': (elu, elu_derivative),
}
#####################################################
choice = 'elu' #funkcja aktywacji do wyboru
#####################################################
activation, derivative = activation_functions[choice]

# forward propagation
def forward_propagation(x, w_hidden, w_output, bias_hidden, bias_output):
    h_pre = bias_hidden + np.dot(w_hidden, x)
    h = activation(h_pre)

    y_pre = bias_output+ np.dot(w_output, h)
    y = tanh(y_pre)
    return h, y

# liczenie MSE
def calculate_mse(y_pred, y_true):
    return np.nanmean((y_pred - y_true) ** 2)

# backpropagation
def backpropagation(x, h, y_pred, y_true, w_hidden, w_output, bias_hidden, bias_output, learning_rate):
    lambda_reg = 0.001

    delta_y = np.clip(y_pred - y_true, -5, 5)
    w_output -= learning_rate * (np.dot(delta_y, h.T) + lambda_reg * w_output)
    bias_output -= learning_rate * np.sum(delta_y, axis=1, keepdims=True)

    delta_h = np.clip(np.dot(w_output.T, delta_y) * derivative(h), -5, 5)
    w_hidden -= learning_rate * (np.dot(delta_h, x.T) + lambda_reg * w_hidden)
    bias_hidden -= learning_rate * np.sum(delta_h, axis=1, keepdims=True)

    return w_hidden, w_output, bias_hidden, bias_output

# trenowanie modelu
def train_model(x_train, y_train, w_hidden, w_output, bias_hidden, bias_output, learning_rate, epochs):
    for epoch in range(epochs):
        h, y_pred = forward_propagation(x_train, w_hidden, w_output, bias_hidden, bias_output)
        mse = calculate_mse(y_pred, y_train)

        w_hidden, w_output, bias_hidden, bias_output = backpropagation(x_train, h, y_pred, y_train, w_hidden, w_output,
                                                                       bias_hidden, bias_output, learning_rate)

        #print(f"Epoch {epoch + 1}, Min w_hidden: {np.min(w_hidden)}, Max w_hidden: {np.max(w_hidden)}")
        #print(f"Epoch {epoch + 1}, Min w_output: {np.min(w_output)}, Max w_output: {np.max(w_output)}")
        print(f"Epoch {epoch + 1}, MSE: {mse:.6f}")
    return w_hidden, w_output, bias_hidden, bias_output

# testowanie modelu w regularnych odstepach
def test_model(w_hidden, w_output, bias_hidden, bias_output, show_actual=True):
    x1_test = np.arange(-2, 2, 0.01)
    x2_test = np.arange(-2, 2, 0.01)
    X1, X2 = np.meshgrid(x1_test, x2_test)
    x_test = np.vstack((X1.ravel(), X2.ravel()))

    _, y_pred = forward_propagation(x_test, w_hidden, w_output, bias_hidden, bias_output)
    y_min, y_max = 0, 20
    y_pred = (y_pred + 1) / 2 * (y_max - y_min) + y_min

    if show_actual:
        y_true = ackley(X1.ravel(), X2.ravel())

    # wspolna skala kolorow, zeby bylo lepiej widoczne
    global_min = min(np.min(y_pred), np.min(y_true) if show_actual else np.min(y_pred))
    global_max = max(np.max(y_pred), np.max(y_true) if show_actual else np.max(y_pred))

    plt.figure(figsize=(10, 10))
    plt.title("Funkcja Ackley'a 2D")

    plt.scatter(x_test[0], x_test[1], c=y_pred.ravel(), cmap='coolwarm', alpha=0.7, vmin=global_min, vmax=global_max,
                label='Aproksymacja')

    if show_actual:
        plt.scatter(x_test[0], x_test[1], c=y_true, cmap='plasma', alpha=0.3, vmin=global_min, vmax=global_max,
                    label='Rzeczywiste wartości')

    plt.colorbar(label="f(x, y)")
    plt.legend()
    #plt.savefig(f'Funkcja_Ackleya_2D_{choice}_0_0001_5.png')
    plt.show()

def plot_ackley_3d():
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = ackley(X1, X2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='k', alpha=0.7)
    ax.set_title("Funkcja Ackley'a 3D")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    #plt.savefig(f'Funkcja_Ackleya_3D_{choice}_0_0001_3000.png')
    plt.show()

# Funkcja badająca zależność learning rate od MSE
def learning_rate_mse(x_train, y_train, input_size, hidden_size, output_size, epochs, learning_rates):
    mse_results = []

    for lr in learning_rates:
        # Inicjalizacja wag i biasów
        w_hidden, w_output, bias_hidden, bias_output = initialize_weights(input_size, hidden_size, output_size)

        # Trenowanie modelu
        for epoch in range(epochs):
            h, y_pred = forward_propagation(x_train, w_hidden, w_output, bias_hidden, bias_output)
            mse = calculate_mse(y_pred, y_train)

            w_hidden, w_output, bias_hidden, bias_output = backpropagation(
                x_train, h, y_pred, y_train, w_hidden, w_output, bias_hidden, bias_output, lr
            )

        # Zapisanie końcowego MSE dla danego learning rate
        mse_results.append(mse)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, mse_results, marker='o', linestyle='-', color='b')
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Zależność learning rate od MSE')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #plt.savefig(f'learning_rate_mse_{choice}_3000.png')
    plt.show()




# funkcja main
if __name__ == "__main__":
    input_size = 2
    hidden_size = 50
    output_size = 1
    learning_rate = 0.0001
    epochs = 3000

    x_train, y_train = generate_data(1000)
    w_hidden, w_output, bias_hidden, bias_output = initialize_weights(input_size, hidden_size, output_size)

    w_hidden, w_output, bias_hidden, bias_output = train_model(x_train, y_train, w_hidden, w_output, bias_hidden,
                                                               bias_output, learning_rate, epochs)

    test_model(w_hidden, w_output, bias_hidden, bias_output)
    plot_ackley_3d()

    # tworzenie wykresu zależności learning_rate od mse
    #learning_rates = np.arange(0.0001, 0.001, 0.0001)
    #learning_rate_mse(x_train, y_train, input_size, hidden_size, output_size, epochs, learning_rates)