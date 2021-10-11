# PyTorchSARproject
## Project Description:
Synthetic Aperture Radar (SAR) is the radar technology used by satellites to capture high resolution images of the Earth’s topography. It also has application in body imaging. One of the advantages of using radar over optical sensor is that it can capture images both during day and night-time. Another obstacle overcome by the radar is that it does not get blocked by cloud cover or other atmospheric artifacts such as smoke or smog that can affect the visibility of objects or land features in the satellite feed. However, one disadvantage of using ISRO’s RISAT satellite is that a high exponential noise is produced in the images which makes it extremely grainy leading to subsequent information loss. Thus, in this project I have proposed a state-of-the-art despeckling module which uses wavelet transform and Principal on Approximation Coefficients (POAC) algorithm to remove noise from SAR Images.


### Images Output Before and After Despeckling [Regionwise Screenshots]:
Noisy or Speckled Images   |  Despeckled or denoised Images
:-------------------------:|:-------------------------:
 ![0_band_1](https://user-images.githubusercontent.com/32778343/136120723-837a05b0-40c4-413d-bbf8-4804ebf6c932.jpg)|![0_band_1 (1)](https://user-images.githubusercontent.com/32778343/136120743-5b7da4a6-554a-4414-bc8e-dc06eff63169.jpg)
  ![1_band_2](https://user-images.githubusercontent.com/32778343/136120749-fb22f616-8dfa-4bb3-b1ac-d67fdbabfe92.jpg)|![1_band_1](https://user-images.githubusercontent.com/32778343/136120768-5cfd0ff0-e720-4258-9a9f-35294f498198.jpg)
![2_band_2](https://user-images.githubusercontent.com/32778343/136120794-25dc2213-e411-4b57-ad6a-a6bf43c3a12f.jpg)|![2_band_1](https://user-images.githubusercontent.com/32778343/136120803-346f67fe-896d-4186-a335-4b3fd51a8ad2.jpg)
![3_band_2](https://user-images.githubusercontent.com/32778343/136120828-3917e69e-78ad-4217-a16c-e97ad5d6909f.jpg)|![3_band_1](https://user-images.githubusercontent.com/32778343/136120841-a68f1c8c-d3ec-42d8-b267-520eb9298bc2.jpg)
![4_band_2](https://user-images.githubusercontent.com/32778343/136120857-de96546f-49ce-4a0f-9113-3a315368f74f.jpg)|![4_band_1](https://user-images.githubusercontent.com/32778343/136120868-2608c2f8-17a2-4fb6-9387-b5114aa9be65.jpg)

## Architecture:
![Blank diagram](https://user-images.githubusercontent.com/32778343/118477648-8c95be80-b72c-11eb-867d-748c040c010c.jpeg)
## How to Run Despeckle Module:
Create two Folders in the project Directory - "Input_Imgaes" and "Output Images". Put your full size tif format images in the "Input_Images" folder. Run the wavelet_poac_module.py file then your output will be produced in the "Output_Images" folder.
## How to Resize Output for Testing:
Often Times due to odd dimensions of images output may have one unit extra height and width. Thus this resize funtion will help to adjust the output image size before testing them for metrics. In the project directory create a "Resized_Images" folder and run the resize.py funtion. 

Note you will have to add the size for the output in the script file, wherein a comment has been mentioned for the same.
## How to Run Metrics Testing:
Inorder to test input and output images in the "Metrics_Calculation" folder, create Two folder "Input" and "Output". Add the Noisy Images in the "Input" folder and the generated Despeckled Images in the Output folder, then via command prompt run "metrics_test.bat". A csv file for the metrics table will be generated in the "Output" folder.

Note: If you are running a low end system(<16GB RAM and <i7 processor) you can comment out the SSIM calculation in the "metrics.py" file as it is highly compute intensive and get the csv file for other metrics.

Source: https://github.com/L4TTiCe/Despeckler_FIS
## How to Convert a .img file to tiff file:
If your dataset consists of images in .img format you can convert them to tif images before running the speckler supressor module. In the "IMG_2_TIF" folder place your .img files. Next create a folder called "out". Now run the "bin_tiff.py" file either in a code editor or command prompt and your converted tif files will be generated in the "out" folder.

Note- Inorder to use this function you will require GDAL to be installed on your system. Follow the link below for installation guide:

https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows
## Results:

For the Reults we have used SAR Images:

Dataset Link - https://drive.google.com/drive/folders/10c5PnxnlY1ucj_SmMMcI_aNhXShaURto?usp=sharing

Inorder to comapre metrics generated from this module we have used traditional descpeckling filters from SNAP sentinel software(lee, forst and lee Sigma):

![image](https://user-images.githubusercontent.com/32778343/118484113-7b50b000-b734-11eb-938a-a8740618383e.png)

Installing PyTorch:

Installation Guide: https://pytorch.org/get-started/locally/
## Tools used:

1. PyTorch - for creating pipelines and performing operations on images:
2. GDAL - to work across different image formats
3. SNAP sentinel software - To view outputs and inferences
4. Python - Script file used to generate metrics for evaluation and convert image formats

Note - The requirements file will work with only anaconda python environments.
