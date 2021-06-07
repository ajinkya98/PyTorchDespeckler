# PyTorchSARproject
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

## Tools used:

1. PyTorch - for creating pipelines and performing operations on images:
    [Installation Guide]:  https://pytorch.org/get-started/locally/
2. GDAL - to work across different image formats
3. SNAP sentinel software - To view outputs and inferences
4. Python - Script file used to generate metrics for evaluation and convert image formats

Note - The requirements file will work with only anaconda python environments.
