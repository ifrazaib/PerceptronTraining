import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_file = "bezdekIris.csv"  
with open(data_file, 'r') as file:
    lines = file.readlines()

X = []
y = []
for line in lines:
    if line.strip():
        elements = line.strip().split(',')
        X.append(elements[:-1])
        y.append(elements[-1])


label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([label_map[label] for label in y])

X = np.array(X, dtype=float)
scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iterations):
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi

    def predict(self, X):
        return np.where(np.dot(X, self.weights) >= 0, 1, 0)


class GradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iterations):
            output = self.sigmoid(np.dot(X, self.weights))
            error = y - output
            adjustment = np.dot(X.T, error * self.sigmoid_derivative(output))
            self.weights += self.learning_rate * adjustment

    def predict(self, X):
        return np.where(self.sigmoid(np.dot(X, self.weights)) >= 0.5, 1, 0)



def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_accuracy = np.mean(model.predict(X_train) == y_train)
    test_accuracy = np.mean(model.predict(X_test) == y_test)
    return train_accuracy, test_accuracy


# 
perceptron = Perceptron()
gradient_descent = GradientDescent()

# Evaluate models
perceptron_train_accuracy, perceptron_test_accuracy = evaluate_model(perceptron, X_train, X_test, y_train, y_test)
gradient_descent_train_accuracy, gradient_descent_test_accuracy = evaluate_model(gradient_descent, X_train, X_test, y_train, y_test)

# Print results
print("Perceptron Train Accuracy:", perceptron_train_accuracy)
print("Perceptron Test Accuracy:", perceptron_test_accuracy)
print("Gradient Descent Train Accuracy:", gradient_descent_train_accuracy)
print("Gradient Descent Test Accuracy:", gradient_descent_test_accuracy)

# Plotting
labels = ['Perceptron Train', 'Perceptron Test', 'Gradient Descent Train', 'Gradient Descent Test']
train_accuracies = [perceptron_train_accuracy, perceptron_test_accuracy, gradient_descent_train_accuracy, gradient_descent_test_accuracy]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, train_accuracies, width, label='Train Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Attach a text label above each bar in rects, displaying its height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)

fig.tight_layout()

plt.show()
