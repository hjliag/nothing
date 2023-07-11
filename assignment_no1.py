# data import and preprocessing 

import numpy as np
import torch
from torchvision import datasets 

trainset = datasets.MNIST(root='./data', train=True, download=True)
testset = datasets.MNIST(root='./data', train=False, download=True)

# indices for train/val splits: train_idx, valid_idx
np.random.seed(0)
val_ratio = .1
train_size = len(trainset)
indices = list(range(train_size))
split_idx = int(np.floor(val_ratio * train_size))
np.random.shuffle(indices)
train_idx, val_idx = indices[split_idx:], indices[:split_idx]

train_data = trainset.data[train_idx].float()/255.
train_labels = trainset.targets[train_idx]
val_data = trainset.data[val_idx].float()/255.
val_labels = trainset.targets[val_idx]
test_data = testset.data.float()/255.
test_labels = testset.targets

# k-NN's performance when the 'k' is 5 

class KNN:
    def __init__(self, k, batch_size=100):
        self.k = k
        self.batch_size = batch_size 

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, X):
        # used batched calculation of the distances for faster computation 
        num_test = X.size(0)
        num_train = self.X_train.size(0)
        distances = torch.zeros(num_test, num_train)

        for i in range(0, num_test, self.batch_size):
            batch_end = min(i + self.batch_size, num_test)
            batch_distances = torch.cdist(X[i:batch_end], self.X_train)
            distances[i:batch_end] = batch_distances

        return distances

    def predict(self, X):
        distances = self.euclidean_distance(X)
        _, indices = torch.topk(distances, self.k, largest=False, dim=1)
        k_nearest_labels = self.y_train[indices]
        predictions = torch.mode(k_nearest_labels, dim=1).values
        return predictions

k = 5
batch_size = 100

classifier = KNN(k, batch_size)

# training
classifier.train(train_data.view(-1, 28 * 28), train_labels)

# validation on validation dataset 
val_predictions = classifier.predict(val_data.view(-1, 28 * 28))
accuracy = torch.mean((val_predictions == val_labels).float())
print(f"validation accuracy: {accuracy.item() * 100}%")

# evaluation on the test dataset 
test_predictions = classifier.predict(test_data.view(-1, 28 * 28))
test_accuracy = torch.mean((test_predictions == test_labels).float())
print(f"test accuracy: {test_accuracy.item() * 100}%")

class KNN:
    def __init__(self, k, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def calculate_distances(self, X):
        if self.distance_metric == "euclidean":
            distances = torch.cdist(X, self.X_train, p=2)
        elif self.distance_metric == "manhattan":
            distances = torch.cdist(X, self.X_train, p=1)
        elif self.distance_metric == "cosine":
            X_normalized = X / torch.norm(X, dim=1, keepdim=True)
            X_train_normalized = self.X_train / torch.norm(self.X_train, dim=1, keepdim=True)
            distances = 1.0 - torch.matmul(X_normalized, X_train_normalized.t())
        else:
          print('no!')
          exit()

        return distances

    def predict(self, X):
        distances = self.calculate_distances(X)
        _, indices = torch.topk(distances, self.k, largest=False, dim=1)
        k_nearest_labels = self.y_train[indices]
        predictions = torch.mode(k_nearest_labels, dim=1).values
        return predictions

k_values = [5, 10, 15]
distance_metrics = ["euclidean", "manhattan", "cosine"]

best_accuracy = 0
best_k = 0
best_distance_metric = ''

for k in k_values:
    for distance_metric in distance_metrics:
        classifier = KNN(k, distance_metric)
        
        # train
        classifier.train(train_data.view(-1, 28 * 28), train_labels)
        
        # validate the hyperparameters
        val_predictions = classifier.predict(val_data.view(-1, 28 * 28))
        accuracy = torch.mean((val_predictions == val_labels).float())
        print(f"validation accuracy (k={k}, distance_metric={distance_metric}): {accuracy.item() * 100}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_distance_metric = distance_metric

# evaluate with the best hyperparameters
best_classifier = KNN(best_k, best_distance_metric)
best_classifier.train(train_data.view(-1, 28 * 28), train_labels)
test_predictions = best_classifier.predict(test_data.view(-1, 28 * 28))
test_accuracy = torch.mean((test_predictions == test_labels).float())
print()
print(f"test accuracy (k={best_k}, distance_metric={best_distance_metric}): {test_accuracy.item() * 100}%")