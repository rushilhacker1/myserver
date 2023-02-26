import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device == "cpu"):
  torch.backends.openvino.enabled = True
  torch.backends.openvino.device = device
print(f"Using device: {device}")

# Define hyperparameters
input_size = 784
hidden_sizes = [500] * 50
output_size = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 20
dropout_prob = 0.5

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Define the model
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(input_size, hidden_sizes[0]),
                      nn.Sigmoid(),
                      nn.Dropout(dropout_prob))
for i in range(1, len(hidden_sizes)):
    model.add_module(f"hidden_layer_{i}", nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
    model.add_module(f"sipoid_activation_{i}", nn.Sigmoid())
    model.add_module(f"dropout_{i}", nn.Dropout(dropout_prob))
model.add_module("output_layer", nn.Linear(hidden_sizes[-1], output_size))
model.add_module("log_softmax", nn.LogSoftmax(dim=1))
model.to(device)

# Define the optimizer and loss function
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

# Test the model
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

correct_count, all_count = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        ps = torch.exp(output)
        probab = list(ps.cpu().numpy())
        predictions = probab.index(max(probab))
        true_labels = labels.cpu().numpy()
        correct_count += (predictions == true_labels).sum().item()
        all_count += labels.size(0)
    
# Print top 5 error rate
top5_error_rate = 1 - correct_count / all_count
print(f"Top 5 error rate: {top5_error_rate*100:.2f}%")
