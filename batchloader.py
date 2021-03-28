# importing libraries
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from PIL import Image
import torch
from pytorch_wavelets import DWTForward,DWTInverse

# setting data loading to GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Calc mean and std across all images for normalization
# mean = 0.0844
# std = 0.0086

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

        return (img_train,img_validation)

# setting the preprocessing for images before loading the batch
transform = transforms.Compose(
        [
            # transforms.Resize(size=32),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean,std=std),
            # transforms.GaussianBlur(kernel_size=5,sigma = 0.1)
            transforms.Grayscale(num_output_channels=1)
        ]
    )

# hyperparameters
batch_size = 4
shuffle = True
num_workers = 2
pin_memory = True

#loading the custom created dataset
dataset = Custom_Dataset("Original","Filtered",transform=transform)
#creating dataloaders for train and validation for the filtered folder images
train_set, validation_set = torch.utils.data.random_split(dataset,[340,85])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
# validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

def wavelet_transform(tensor):
    """ Wavelet transform function to extract ll,lh,hl,ll band images from SAR satellite images """
    transformer = DWTForward(J=1, mode='symmetric', wave='db3')
    ll_band_image,low_band = transformer(tensor)  #ll_band_image --> [4,3,514,514], low_band --> list of deconstruction tensors, for level 1(single tensor) --> [4,1,3,514,514]
    return ll_band_image,low_band[0][:,:,0,:,:],low_band[0][:,:,1,:,:],low_band[0][:,:,2,:,:]

def poac(ll,lh,hl,hh):
    """ Perform POAC to get denoised images """
    slh = torch.trace(torch.matmul(ll,torch.transpose(lh,dim0=0,dim1=1)))/torch.trace(torch.matmul(ll,torch.transpose(ll,dim0=0,dim1=1))) # slh=(trace(A*B')/trace(A*A'))
    shl = torch.trace(torch.matmul(ll, torch.transpose(hl,dim0=0,dim1=1))) / torch.trace(torch.matmul(ll, torch.transpose(ll,dim0=0,dim1=1))) # shl=(trace(A*C')/trace(A*A'))
    shh = torch.trace(torch.matmul(ll, torch.transpose(hh,dim0=0,dim1=1))) / torch.trace(torch.matmul(ll, torch.transpose(ll,dim0=0,dim1=1))) # shh=(trace(A*D')/trace(A*A'))
    bdash = slh*ll  # Bdash=slh*A
    cdash = shl*ll  # Cdash=shl*A
    ddash = shh*ll  # Ddash=shh*A
    return torch.cat((torch.unsqueeze(bdash,0),torch.unsqueeze(cdash,0),torch.unsqueeze(ddash,0)),dim=0)

def wavelet_inverse(ll,lowband):
    """ Waevlet Inverse to get the reconstructed image after denoising """
    transformer = DWTInverse(wave="db3", mode="symmetric")
    return transformer((ll,[lowband]))

# main data loading and training loop
if __name__ == '__main__':
    for (train_imgs, clean_imgs) in train_loader:
        ll,lh,hl,hh = wavelet_transform(train_imgs)
        list_poac_imgs = []
        for i in range(batch_size):
            list_poac_imgs.append(poac(torch.squeeze(ll[i,:,:,:]),torch.squeeze(lh[i,:,:,:]),torch.squeeze(hl[i,:,:,:]),torch.squeeze(hh[i,:,:,:])))
        jdash = torch.cat((torch.unsqueeze(torch.unsqueeze(list_poac_imgs[0],0),0),torch.unsqueeze(torch.unsqueeze(list_poac_imgs[1],0),0),torch.unsqueeze(torch.unsqueeze(list_poac_imgs[2],0),0),torch.unsqueeze(torch.unsqueeze(list_poac_imgs[3],0),0)),dim=0)
        train_img_poac=wavelet_inverse(ll,jdash) # Jdash=idwt2(A,Bdash,Cdash,Ddash,'coif3')
        print(train_img_poac.size())  # sanity check to ensure image size is intact after reconstruction
        train_img_poac = train_img_poac.to(device)  # moving data to GPU
        clean_imgs = clean_imgs.to(device)



