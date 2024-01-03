import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import torch.nn as nn
from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare the data
images = []
labels = []
dataPath = r'C:\Users\Admin\Desktop\CatDog Data'
subFolder = os.listdir(dataPath)

# Iterate through subfolders (assuming each subfolder represents a class)
for folder in subFolder:
    label = subFolder.index(folder)
    path = os.path.join(dataPath, folder)
    for imglist in os.listdir(path):
        image = cv2.imread(os.path.join(path, imglist))
        images.append(image)
        labels.append(label)

# Define a custom Dataset
class DataPrep(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        image = self.features[item]
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)

# Define the transformation for the input images
data_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((200, 200))
])

# Create the dataset and data loader
dataset = DataPrep(images, labels, data_trans)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
data_sample = next(iter(data_loader))

# Define the CatDogModel
class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(64 * 50 * 50, 2)
        self.flatten = nn.Flatten()

    def forward(self, data):
        data = self.conv1(data)
        data = self.activation(data)
        data = self.maxpool(data)
        data = self.activation(data)
        data = self.flatten(data)
        data = self.linear(data)
        return data

# Create an instance of the model and move it to the device
model = CatDogModel().to(device)
print(model.eval())

# Define the training parameters
learning_rate = 0.001
epoch = 100
optimizer = optim.SGD(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

total_correct = 0
total_samples = 0

# Training loop
for i in range(epoch):
    for image, target in data_loader:
        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(image)
        _, predicted = torch.max(output, 1)

        total_samples += target.size(0)
        total_correct += (predicted == target).sum().item()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    accuracy = total_correct / total_samples
    print(f'Epochs: {i+1} out of {epoch} || Loss: {loss.item()} || Accuracy: {accuracy}')

# Save the trained model
torch.save(model.state_dict(), 'catdog_model.pt')
print('Model saved')

# Load the saved model
loaded_model = CatDogModel()
loaded_model.load_state_dict(torch.load('catdog_model.pt'))
loaded_model = loaded_model.to(device)

# Load and preprocess an image for inference
image_path = r'C:\Users\Admin\Desktop\cats/cat.36.jpg'
image = cv2.imread(image_path)
input_image = data_trans(image).unsqueeze(0).to(device)

# Run inference on the loaded model
with torch.no_grad():
    output = loaded_model(input_image)

_, predicted_class = torch.max(output.data, 1)
class_label_map = {0: 'Cat', 1: 'Dog'}
predicted_label = class_label_map[predicted_class.item()]
print('Predicted label:', predicted_label)