from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from PIL import Image
import torch

# setting data loading to GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None
# Calc mean and std across all images for normalization
# mean = 0.0844
# std = 0.0086

#creating custom dataloader for training and validation images
class Custom_Dataset(Dataset):
    def __init__(self, train_dir,transform=None):

        self.train_dir = train_dir
        self.transform = transform

    def __len__(self):

        return len(os.listdir(self.train_dir))

    def __getitem__(self, index):

        img_train_id = os.listdir(self.train_dir)[index]
        img_train = Image.open(os.path.join(self.train_dir, img_train_id))

        if self.transform is not None:
            img_train = self.transform(img_train)

        return img_train

# setting the preprocessing for images before loading the batch
transform = transforms.Compose(
        [
            transforms.Resize(size=(13974,13464)), #enter the size you want eg is put here
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean,std=std),
            # transforms.GaussianBlur(kernel_size=5,sigma = 0.1)
            # transforms.Grayscale(num_output_channels=1)
        ]
    )

# hyperparameters
batch_size = 1
shuffle = False
num_workers = 2
pin_memory = True

#loading the custom created dataset
dataset = Custom_Dataset("Output_Images",transform=transform)
# #creating dataloaders for train and validation for the filtered folder images
# train_set, validation_set = torch.utils.data.random_split(dataset,[5,0])
train_loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
# validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

if __name__ == '__main__':
    j = 4
    for train_imgs in train_loader:
        print(train_imgs.size())
        img_numpy = torch.squeeze(torch.squeeze(train_imgs,dim=0).permute(1,2,0),dim=2).numpy()
        print(img_numpy.shape)
        img_final = Image.fromarray(img_numpy)
        img_final.save("Resized_Images/"+str(j)+".tif")
