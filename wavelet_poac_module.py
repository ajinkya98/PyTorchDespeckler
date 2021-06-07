# importing libraries
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from PIL import Image
import torch
from pytorch_wavelets import DWTForward,DWTInverse
from pytictoc import TicToc

# setting data loading to GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None # Overrides the limit of the size of Image that can be input by PIL
t = TicToc()


#creating custom dataloader for training and validation images
class Custom_Dataset(Dataset):
    def __init__(self, train_dir,transform=None):

        self.train_dir = train_dir  #directory where input images are kept
        self.transform = transform  #transform operation to convert image to tensor

    def __len__(self):

        return len(os.listdir(self.train_dir))   # get total number of images in the input_images folder

    def __getitem__(self, index):

        img_train_id = os.listdir(self.train_dir)[index]  # indexing imput images in the folder
        img_train = Image.open(os.path.join(self.train_dir, img_train_id))  # opening image using PIL

        if self.transform is not None:
            img_train = self.transform(img_train)   # tranform func to convert image to tensor

        return img_train

# setting the preprocessing for images before loading the batch
transform = transforms.Compose(
        [
            # transforms.Resize(size=(10000,10000)),   # only tensor conversion, other pre-proccessing techniques not used
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean,std=std),
            # transforms.GaussianBlur(kernel_size=5,sigma = 0.1)
        ]
    )

# hyperparameters
batch_size = 1 # takes in a single image for computation
shuffle = True # setting this to true will shuffle the order in which input images load
num_workers = 2 # number of parallel proccessing units often to do with low level performance, set to 2 or more in code editor for jupyter nb set to 0
pin_memory = True # If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, you can speed up the host to device transfer by enabling pin_memory

#loading the custom created dataset
dataset = Custom_Dataset("./Input_Images",transform=transform) #calling the created custom dataset
train_loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory) # calling the dataset loader

def wavelet_transform(tensor): # Check pytorch wavelets doc for more info
    """ Wavelet transform function to extract ll,lh,hl,ll band images from SAR satellite images """
    transformer = DWTForward(J=1, mode='symmetric', wave='db3')
    ll_band_image,low_band = transformer(tensor)  
    return ll_band_image,low_band[0][:,:,0,:,:],low_band[0][:,:,1,:,:],low_band[0][:,:,2,:,:] #images are concatenated thus indexing to unload from a single tensor

def poac(ll,lh,hl,hh):
    """ Perform POAC to get denoised images """
    slh = torch.trace(torch.matmul(ll,torch.transpose(lh,dim0=0,dim1=1)))/torch.trace(torch.matmul(ll,torch.transpose(ll,dim0=0,dim1=1))) # slh=(trace(A*B')/trace(A*A'))
    shl = torch.trace(torch.matmul(ll, torch.transpose(hl,dim0=0,dim1=1))) / torch.trace(torch.matmul(ll, torch.transpose(ll,dim0=0,dim1=1))) # shl=(trace(A*C')/trace(A*A'))
    shh = torch.trace(torch.matmul(ll, torch.transpose(hh,dim0=0,dim1=1))) / torch.trace(torch.matmul(ll, torch.transpose(ll,dim0=0,dim1=1))) # shh=(trace(A*D')/trace(A*A'))
    bdash = slh*ll  # Bdash=slh*A
    cdash = shl*ll  # Cdash=slh*A
    ddash = shh*ll  # Ddash=shh*A
    return torch.cat((torch.unsqueeze(bdash,0),torch.unsqueeze(cdash,0),torch.unsqueeze(ddash,0)),dim=0) # concatenate three sub-bands for IDWT func

def wavelet_inverse(ll,lowband):
    """ Wavelet Inverse to get the reconstructed image after denoising """
    transformer = DWTInverse(wave="db3", mode="symmetric")
    return transformer((ll,[lowband]))

# main data loading and training loop
if __name__ == '__main__':
    j = 0 #indexing input images in the folder
    for train_imgs in train_loader:
        t.tic() # start calc. time to measure computation time
        print(str(j)+":") #printing info about original image
        print("Original Image Dimension:")
        print(train_imgs.size())
        print(train_imgs.dtype)
        print(train_imgs)
        # train_imgs = torch.nan_to_num(train_imgs) #if images have any dead pixels or nan values this can be used
        ll,lh,hl,hh = wavelet_transform(train_imgs) #wavelet transform to get sub-band images
        list_poac_imgs = []   #creating empty list for appending sub-band images
        for i in range(batch_size):
            list_poac_imgs.append(poac(torch.squeeze(ll[i,:,:,:]),torch.squeeze(lh[i,:,:,:]),torch.squeeze(hl[i,:,:,:]),torch.squeeze(hh[i,:,:,:])))
        jdash = torch.unsqueeze(torch.unsqueeze(list_poac_imgs[0],0),0)  #adding extra dim for adjusting for idwt func
        train_img_poac=wavelet_inverse(ll,jdash) # Jdash=idwt2(A,Bdash,Cdash,Ddash,'coif3')
        print("Image Dimensions After wavelet and POAC transform:")
        print(train_img_poac.size())  # sanity check to ensure image size is intact after reconstruction
        img_numpy = torch.squeeze(torch.squeeze(train_img_poac,dim=0).permute(1,2,0),dim=2).numpy() #converting tensor to numpy array for saving output image. Pytorch tensors and numpy arrays have diff channel arrangement.
        t.toc() #stop to cal. computation time
        print("Numpy converted Dimension:") #printing numpy converted dimensions
        print(img_numpy.shape)
        print(img_numpy.dtype)
        print("-----------------------------------------------------------------------------------------")
        img_final = Image.fromarray(img_numpy) # saving image using PIL
        img_final.save("Output_Images/"+str(j)+".tif") #Tif format for saving the output despeckled image.
        j+=1



