import torch
from torch.utils import data
from torchvision import transforms as T
from mnist import MNIST
import numpy as np
from PIL import Image
#from skimage import color
from torch.utils.data import WeightedRandomSampler


class MNISTData(data.Dataset):
    """Dataset class for the MNIST dataset."""

    def __init__(self, data_path, train, transform):
        self.mndata = MNIST(data_path)
        self.train = train
        self.transform = transform
        self.preprocess()        
     
 
    def preprocess(self):
        if self.train: 
            images, labels = self.mndata.load_training()
        else:
            images, labels = self.mndata.load_testing()
   
        self.images = images
        self.labels = labels


    def __getitem__(self, index):
        I = self.images[index] 
        I = np.reshape(np.array(I), (28,28,1))
        I = np.pad(I, ((2,2), (2,2), (0,0)), mode='constant', constant_values=(0,))
        I = I/255.0
        I = np.transpose(I, (2, 0, 1))
        #I = Image.fromarray(I.astype(np.uint8), mode='RGB')
        return torch.FloatTensor(I) #self.transform(I)
 

    def __len__(self):
        return len(self.images)


def get_loader(batch_size, im_size, train):

    transforms = []
    #transforms.append(T.Resize((im_size, im_size)))
    transforms.append(T.ToTensor())
    transforms = T.Compose(transforms)

    path = '../data/python-mnist/data' 
    dataset = MNISTData(path, train, transforms)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size)
    return data_loader
