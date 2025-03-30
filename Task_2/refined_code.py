import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, confusion_matrix


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: expit(x)
        self.inverse_activation_function = lambda x: logit(x)

    def fit(self, X, y, epochs=5):
        for e in range(epochs):
            for i in range(len(X)):
                self.train(X[i], y[i])

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = self.wih @ inputs
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = self.who @ hidden_outputs
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = self.who.T @ output_errors

        self.who += self.lr * (output_errors * final_outputs * (1.0 - final_outputs)) @ hidden_outputs.T
        self.wih += self.lr * (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)) @ inputs.T

    def predict(self, X):
        y_pred = []
        for inputs_list in X:
            inputs = np.array(inputs_list, ndmin=2).T
            hidden_outputs = self.activation_function(self.wih @ inputs)
            final_outputs = self.activation_function(self.who @ hidden_outputs)
            y_pred.append(np.argmax(final_outputs))
        return np.array(y_pred)


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()

    X, y = [], []
    for line in data:
        all_values = line.strip().split(',')
        X.append((np.asarray(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01)
        target = np.full(10, 0.01, dtype=np.float32)
        target[int(all_values[0])] = 0.99
        y.append(target)

    labels = [int(line.strip().split(',')[0]) for line in data]
    return np.array(X), np.array(y), np.array(labels)


def evaluate_model(model, X_test, y_test_labels):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test_labels, predictions)
    conf = confusion_matrix(y_test_labels, predictions)
    return acc, conf, predictions


def run_experiments():
    X_train, y_train, _ = load_data("./Day7NN/Day7NN/mnist_train.csv")
    X_test, y_test, y_test_labels = load_data("./Day7NN/Day7NN/mnist_test.csv")

    hidden_nodes_options = [50, 200]
    learning_rates = np.round(np.power(10, np.linspace(-3, 0, 15)) * 1000) / 1000

    results = {}

    for hidden_nodes in hidden_nodes_options:
        scores = []
        for lr in learning_rates:
            model = NeuralNetwork(784, hidden_nodes, 10, lr)
            model.fit(X_train[:1000], y_train[:1000], epochs=1)
            acc, _, _ = evaluate_model(model, X_test[:200], y_test_labels[:200])
            scores.append(acc)
        results[hidden_nodes] = scores

    for hidden_nodes in hidden_nodes_options:
        plt.plot(learning_rates, results[hidden_nodes], label=f"Hidden: {hidden_nodes}")

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("HariSudan_ID_plot2.png")
    plt.show()


if __name__ == '__main__':
    run_experiments()