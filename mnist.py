import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

#funkcja do wczytywania danych
def load_data(train_images_file, train_labels_file, test_images_file, test_labels_file):
    train_images = idx2numpy.convert_from_file(train_images_file)  # (60000, 28, 28)
    train_labels = idx2numpy.convert_from_file(train_labels_file)
    test_images = idx2numpy.convert_from_file(test_images_file)
    test_labels = idx2numpy.convert_from_file(test_labels_file)
    return train_images, train_labels, test_images, test_labels

#funkcja do konwersji na one-hot encoding
def one_hot_encoding(labels, num_classes):
    return np.eye(num_classes)[labels]

#funkcja do spłaszczania obrazów
def flatten(images):
    return images.reshape(images.shape[0], -1)

#funkcja do normalizacji danych do zakresu [0-1]
def normalize(images):
    return images / 255


#funkcja do inicjalizacji wag i biasów
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

#softmax
def softmax(z):
    z_stabilized = z - np.max(z)
    exp_z = np.exp(z_stabilized)
    a = exp_z / np.sum(exp_z)
    return a

activation_functions = {
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
}
###################################################
choice = 'leaky_relu' #funkcja aktywacji do wyboru
###################################################
activation, derivative = activation_functions[choice]

#funkcja do obliczania forward propagation
def forward_propagation(img, w_hidden, w_output, bias_hidden, bias_output):
    h_pre = bias_hidden + np.dot(w_hidden, img)
    h = activation(h_pre)

    o_pre = bias_output + np.dot(w_output, h)
    o = softmax(o_pre)
    return h, o

#funkcja do obliczania średniokwadratowego błędu
def calculate_mse(o, l):
    return np.mean((o - l) ** 2)

#funkcja do ewaluacji modelu na zbiorze testowym
def evaluate_model(test_images, test_labels, w_hidden, w_output, bias_hidden, bias_output):
    correct_predictions = 0
    total_mse = 0

    for img, label in zip(test_images, test_labels):
        img = img.reshape(-1, 1)  # Zamiana wektora na macierz
        label = label.reshape(-1, 1)  # Zamiana wektora na macierz

        _, o = forward_propagation(img, w_hidden, w_output, bias_hidden, bias_output)
        total_mse += calculate_mse(o, label)
        correct_predictions += int(np.argmax(o) == np.argmax(label))

    accuracy = (correct_predictions / test_images.shape[0]) * 100
    average_mse = total_mse / test_images.shape[0]

    return accuracy, average_mse

#funkcja do obliczania back propagation
def backpropagation(img, h, o, label, w_hidden, w_output, bias_hidden, bias_output, learning_rate, lambda_reg = 0.001):
    delta_o = o - label
    w_output += -learning_rate * (np.dot(delta_o, np.transpose(h)) + lambda_reg * w_output)
    bias_output += -learning_rate * delta_o

    delta_h = np.dot(np.transpose(w_output), delta_o) * derivative(h)
    w_hidden += -learning_rate * (np.dot(delta_h, np.transpose(img)) + lambda_reg * w_hidden)
    bias_hidden += -learning_rate * delta_h

    return w_hidden, w_output, bias_hidden, bias_output

#funkcja do trenowania modelu
def train_model(train_images, train_labels, w_hidden, w_output, bias_hidden, bias_output, learning_rate, epochs):
    correct_predictions = 0
    for epoch in range(epochs):
        for img, label in zip(train_images, train_labels):
            img = img.reshape(-1, 1) #zamiana wektora na macierz
            label = label.reshape(-1, 1) #zamiana wektora na macierz

            h, o = forward_propagation(img, w_hidden, w_output, bias_hidden, bias_output)
            #error = calculate_error(o, label)
            correct_predictions += int(np.argmax(o) == np.argmax(label)) #ilość poprawnie sklasyfikowanych liczb

            w_hidden, w_output, bias_hidden, bias_output = backpropagation(img, h, o, label, w_hidden, w_output, bias_hidden, bias_output, learning_rate)

        print(f"Accuracy after epoch {epoch + 1}: {round((correct_predictions / train_images.shape[0]) * 100, 2)}%")
        #print(f"Active neurons in hidden layer: {np.sum(h > 0)} out of {h.size}")
        correct_predictions = 0

    return w_hidden, w_output, bias_hidden, bias_output


# Funkcja do testowania różnych współczynników uczenia
def test_learning_rates(train_images, train_labels, test_images, test_labels, input_size, hidden_size, output_size,
                        epochs):
    learning_rates = np.arange(0.0001, 0.002, 0.0001)  # dla ilu roznych learning rates wytrenowac w zakresie
    mse_values = []  # Lista na wartości MSE dla każdego learning_rate

    for lr in learning_rates:
        # Inicjalizacja wag i biasów
        w_hidden, w_output, bias_hidden, bias_output = initialize_weights(input_size, hidden_size, output_size)

        # Trenowanie modelu
        w_hidden_opt, w_output_opt, bias_hidden_opt, bias_output_opt = train_model(
            train_images, train_labels, w_hidden, w_output, bias_hidden, bias_output, lr, epochs
        )

        # Ewaluacja modelu na danych testowych
        _, mse = evaluate_model(test_images, test_labels, w_hidden_opt, w_output_opt, bias_hidden_opt, bias_output_opt)
        mse_values.append(mse)

    # Rysowanie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, mse_values, marker='o', color='b')
    plt.title(f"Zależność stopnia uczenia od błędu średniokwadratowego dla {epochs} przejść uczących przez dane")
    plt.xlabel("Learning Rate")
    plt.ylabel("MSE")
    plt.grid()
    plt.savefig(f"Zależność stopnia uczenia od błędu średniokwadratowego dla {epochs} przejść uczących przez dane")
    plt.show()

    return learning_rates, mse_values


def main():
    #ścieżki do plików
    train_images_file = "C:\\Users\\Oskar\\Desktop\\archive\\train-images.idx3-ubyte"
    train_labels_file = "C:\\Users\\Oskar\\Desktop\\archive\\train-labels.idx1-ubyte"
    test_images_file = "C:\\Users\\Oskar\\Desktop\\archive\\t10k-images.idx3-ubyte"
    test_labels_file = "C:\\Users\\Oskar\\Desktop\\archive\\t10k-labels.idx1-ubyte"

    #wczytywanie danych
    train_images, train_labels, test_images, test_labels = load_data(train_images_file, train_labels_file, test_images_file, test_labels_file)

    #parametry uczenia
    input_size = 784 #ilość neuronów warstwy wejściowej
    hidden_size = 20  #ilość neuronów warstwy ukrytej
    output_size = 10 #ilość neuronów warstwy wyjściowej
    learning_rate = 0.0001 #szybkość uczenia
    epochs = 3 #ilość przejść przez dane

    #konwersje danych
    train_labels = one_hot_encoding(train_labels, output_size)
    test_labels = one_hot_encoding(test_labels, output_size)
    train_images = flatten(train_images)
    test_images = flatten(test_images)
    train_images = normalize(train_images)
    test_images = normalize(test_images)


    #inicjalizacja wag i biasów
    w_hidden, w_output, bias_hidden, bias_output = initialize_weights(input_size, hidden_size, output_size)

    #trenowanie modelu
    w_hidden_opt, w_output_opt, bias_hidden_opt, bias_output_opt = train_model(train_images, train_labels, w_hidden, w_output, bias_hidden, bias_output, learning_rate, epochs)


    # ewaluacja modelu na danych testowych
    accuracy, mse = evaluate_model(test_images, test_labels, w_hidden_opt, w_output_opt, bias_hidden_opt, bias_output_opt)
    print(f"Test Accuracy: {round(accuracy, 2)}%")
    print(f"Mean Squared Error on Test Set: {mse}")

    learning_rates, mse_values = test_learning_rates(
        train_images, train_labels, test_images, test_labels,
        input_size, hidden_size, output_size, epochs
    )
    # Wyświetlenie wyników
    for lr, mse in zip(learning_rates, mse_values):
        print(f"Learning Rate: {lr:.3f}, MSE: {mse:.4f}")

if __name__ == "__main__":
    main()