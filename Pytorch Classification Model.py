import cv2
import torch.cuda
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os
import torch.nn as nn
from torch import optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'

images = []
labels = []
dataPath = r'C:\Users\Admin\Desktop\CatDog Data'
subFolder = os.listdir(dataPath)
print(subFolder)
for folder in subFolder:
    label = subFolder.index(folder)
    path = os.path.join(dataPath,folder)
    for imglist in os.listdir(path):
        image = cv2.imread(os.path.join(path,imglist))
        images.append(image)
        labels.append(label)

class DataPrep(Dataset):
    def __init__(self,features,labels,transform = None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        image = self.features[item]
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)

        return image , label
    def __len__(self):
        return len(self.labels)

data_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((200,200))
])

dataset = DataPrep(images,labels,data_trans)
data_loader = DataLoader(dataset,batch_size=4,shuffle=True)
data_sample = next(iter(data_loader))


class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2,stride=2,padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(64*50*50,2)
        self.flatten = nn.Flatten()

    def forward(self,data):
        data = self.conv1(data)
        data = self.activation(data)
        data = self.maxpool(data)
        data = self.activation(data)
        data = self.flatten(data)
        data = self.linear(data)
        return data
print(f'The data is : {CatDogModel}')
class Test(nn.MSELoss):
    def __init__(self):
        super(Test,self).__init__()


model = CatDogModel().to(device)
print(model.eval())

learning_rate = 0.001
epochs = 100
optimizer = optim.SGD(model.parameters(),learning_rate)
criterion = nn.CrossEntropyLoss()

total_correct = 0
total_samples = 0

# for i in range(epochs):
#     for image,target in data_loader:
#         image = image.to(device)
#         target = target.to(device)
#         optimizer.zero_grad()
#         output = model(image)
#
#         _, predicted = torch.max(output, 1)
#         total_samples += target.size(0)
#         total_correct += (predicted == target).sum().item()
#
#         loss = criterion(output,target)
#         loss.backward()
#         optimizer.step()
#
#     accuracy = total_correct / total_samples
#     print(f'Epochs: {i+1} out of {epochs} || Loss: {loss.item()} || Accuracy: {accuracy}')
#

# Saving and Testing the Model
# Save the trained model
torch.save(model.state_dict(), 'catdog_model.pt')
print("Model saved.")

# Load the saved model
loaded_model = CatDogModel()
loaded_model.load_state_dict(torch.load('catdog_model.pt'))
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Read a single image using OpenCV
image_path =  r'C:\Users\Admin\Desktop\dogs/dog.10.jpg'
image = cv2.imread(image_path)

# Apply the transformation to the image
input_image = data_trans(image).unsqueeze(0).to(device)

# Pass the input image through the loaded model
with torch.no_grad():
    output = loaded_model(input_image)
# Get the predicted class
_, predicted_class = torch.max(output.data, 1)

# Map the predicted class index to the corresponding label
class_label_map = {0: 'Cat', 1: 'Dog'}
predicted_label = class_label_map[predicted_class.item()]
print('Predicted label:', predicted_label)
