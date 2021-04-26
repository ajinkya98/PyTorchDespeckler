import numpy as np
import math
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

def get_item(train_dir,output_dir,index):
    img_train_id = os.listdir(train_dir)[index]
    img_train = Image.open(os.path.join(train_dir, img_train_id))
    print(str(os.path.join(train_dir, img_train_id)))
    img_output_id = os.listdir(output_dir)[index]
    img_output = Image.open(os.path.join(output_dir, img_output_id))
    print(str(os.path.join(output_dir, img_output_id)))

    return (np.asarray(img_train),np.asarray(img_output))

# Function for Peak Signal to Noise Ratio
def PSNR(denoised_img_array, noisy_img_array):
    mse = np.mean(np.square(np.subtract(denoised_img_array, noisy_img_array)))
    max_pixel_val = 255
    psnr = 10 * np.log((max_pixel_val ** 2) / mse)

    return psnr


# Function for Speckle Suppression Index
def SSI(denoised_img_array, noisy_img_array):
    var_initial = np.var(noisy_img_array)
    var_final = np.var(denoised_img_array)
    mean_initial = np.mean(noisy_img_array)
    mean_final = np.mean(denoised_img_array)

    ssi = (math.sqrt(var_final) * mean_initial) / (mean_final * math.sqrt(var_initial) + 0.000001)

    return ssi


# Function for Equivalent Number of Looks
def ENL(denoised_img_array):
    mean_final = np.mean(denoised_img_array)
    sd_final = np.std(denoised_img_array)

    enl = np.square(mean_final / (sd_final + 1e-7))

    return enl


# Function for Speckle Suppression and Mean Preservation Index
def SMPI(denoised_img_array, noisy_img_array):
    var_initial = np.var(noisy_img_array)
    var_final = np.var(denoised_img_array)
    mean_initial = np.mean(noisy_img_array)
    mean_final = np.mean(denoised_img_array)

    Q = 1 + abs(mean_initial - mean_final)

    smpi = (Q * math.sqrt(var_final)) / math.sqrt(var_initial) + 0.000001

    return smpi


train_dir = "Tiff_images"
output_dir = "Final_Output"

for i in range(len(os.listdir(train_dir))):
    x,y = get_item(train_dir,output_dir,i)
    print(SSI(x,y))


