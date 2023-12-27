import cv2
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os


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

