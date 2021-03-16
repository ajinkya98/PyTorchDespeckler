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

#creating custom dataloader for training and validation images
class Custom_Dataset(Dataset):
    def __init__(self, train_dir,validation_dir,transform=None):

        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.transform = transform

    def __len__(self):

        return len(os.listdir(self.train_dir))

    def __getitem__(self, index):

        img_train_id = os.listdir(self.train_dir)[index]
        img_validation_id = os.listdir(self.validation_dir)[index]
        img_train = Image.open(os.path.join(self.train_dir, img_train_id)).convert("RGB")
        img_validation = Image.open(os.path.join(self.validation_dir, img_validation_id)).convert("RGB")

        if self.transform is not None:
            img_train = self.transform(img_train)
            img_validation = self.transform(img_validation)
        g = transforms.Grayscale(num_output_channels=1)
        return (g(img_train/255),g(img_validation/255))

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
dataset = Custom_Dataset("Original","Filtered",transform=transform)
#creating dataloaders for train and validation for the filtered folder images
train_set, validation_set = torch.utils.data.random_split(dataset,[340,85])
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=4,num_workers=2,pin_memory=True)
# validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

# loading the batch-wise data and storing it in GPU for testing
if __name__ == '__main__':
    for i in train_loader:
        print(i[0].size(),i[1].size())