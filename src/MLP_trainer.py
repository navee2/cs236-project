# code references https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')


# hyperparameters
input_size = 784
batch_size = 64
hidden_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 10
model_num=0

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=False,
                                          transform=transforms.ToTensor())


# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_trained_model():
    model = MLP(input_size, hidden_size, num_classes).to(device)

    # optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Model training
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = ce_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Step [{i + 1}/{total_steps}], Loss:{loss:.4f}')
    return model

def evaluate_model(model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct/total}%')
    return 100 * correct/total

if __name__=="__main__":
    for i in range(30):
        model = get_trained_model()
        torch.save(model.state_dict(), f'../model/model_{i}.ckpt')
        
    #calculate the accuracies
    accuracies = []
    for i in range(30):
        model = MLP(input_size, hidden_size, num_classes).to(device)
        model.load_state_dict(torch.load(f'../model/model_{i}.ckpt'))
        accuracy = evaluate_model(model)
        accuracies.append(accuracy)
    df = pd.DataFrame({'accuracy':accuracies})
    df.to_csv('accuracy.csv')