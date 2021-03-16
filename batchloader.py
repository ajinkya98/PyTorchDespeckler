# importing libraries
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from PIL import Image
import torch

# setting data loading to GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Calc mean and std across all images for normalization
mean = 0.0844
std = 0.0086

#creating custom dataloader for training images
class Train_Filtered(Dataset):
    def __init__(self, root_dir, transform=None):
        """" This Function sets the root directory to load images from it for training. It also sets the transform for preprocessing the images """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """ This function finds the total number of items to iterate through """
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        """ This function provides instructions for fetching a single item from the dataset which can then be extended to a batch """
        img_id = os.listdir(self.root_dir)[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        g = transforms.Grayscale(num_output_channels=1)
        return g(img/255)
# setting the preprocessing for images before loading the batch
transform = transforms.Compose(
        [
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
            transforms.GaussianBlur(kernel_size=5,sigma = 0.1)
        ]
    )

#loading the custom created dataset
dataset = Train_Filtered("Filtered",transform=transform)
#creating dataloaders for train and validation for the filtered folder images
train_set, validation_set = torch.utils.data.random_split(dataset,[340,85])
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=4,num_workers=2,pin_memory=True)
# validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

# loading the batch-wise data and storing it in GPU for testing
if __name__ == '__main__':
    for i in train_loader:
        print(i.size())