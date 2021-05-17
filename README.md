# PyTorchSARproject
![Blank diagram](https://user-images.githubusercontent.com/32778343/118477648-8c95be80-b72c-11eb-867d-748c040c010c.jpeg)
## How to Run Despeckle Module:
Create two Folders in the project Directory - "Input_Imgaes" and "Output Images". Put your full size tif format images in the "Input_Images" folder. Run the wavelet_poac_module.py file then your output will be produced in the "Output_Images" folder.
## How to Resize Output for Testing:
Often Times due to odd dimensions of images output may have one unit extra height and width. Thus this resize funtion will help to adjust the output image size before testing them for metrics. In the project directory create a "Resized_Images" folder and run the resize.py funtion. Note you will have to add the size for the output in the script file, wherein a comment has been mentioned for the same.
## How to Run Metrics Testing:
Inorder to test input and output images in the "Metrics_Calculation" folder, create Two folder "Input" and "Output". Add the Noisy Images in the "Input" folder and the generated Despeckled Images in the Output folder, then via command prompt run "metrics_test.bat". A csv file for the metrics table will be generated in the "Output" folder.
Note: If you are running a low end system(<16GB RAM and <i7 processor) you can comment out the SSIM calculation in the "metrics.py" file as it is highly compute intensive and get the csv file for other metrics.
## How to Convert a .img file to tiff file:
If your dataset consists of images in .img format you can convert them to tif images before running the speckler supressor module. In the "IMG_2_TIF" folder place your .img files. Next create a folder called "out". Now run the "bin_tiff.py" file either in a code editor or command prompt and your converted tif files will be generated in the "out" folder.
## Results:
For the Reults we have used SAR Images:
Dataset Link - https://drive.google.com/drive/folders/10c5PnxnlY1ucj_SmMMcI_aNhXShaURto?usp=sharing
Inorder to comapre metrics generated from this module we have used traditional descpeckling filters from SNAP sentinel software(lee, forst and lee Sigma):

Evaluation Metrics For Wavelet + POAC Module:						
Name	PSNR	SSI	SMPI	SSIM	MEAN_in	MEAN_out
0.tif	44.80854014	0.865990877	0.88123351	0.972078103	768.9508667	768.9684448
1.tif	43.92271565	0.856835425	0.864567515	0.966692393	994.5228271	994.5137939
2.tif	34.24091108	0.895878673	0.920366415	0.863256233	2816.346924	2816.31958
3.tif	43.56934071	0.952216804	0.970794856	0.964477771	949.2778931	949.2583618
4.tif	38.37904793	0.977816403	1.006922465	0.940058399	1635.908691	1635.878906
						
Evalutaion Metrics for SNAP Speckle Filters:						
Frost:						
Name	PSNR	SSI	SMPI	SSIM	MEAN_in	MEAN_out
0.tif	43.22769836	0.719175041	2.162859428	0.959922455	768.9508667	766.9355469
1.tif	41.25285957	0.72042352	1.992166973	0.941336932	994.5228271	992.7526245
2.tif	31.10448643	0.77870971	8.97006264	0.790807248	2816.346924	2805.784424
3.tif	43.69540513	0.754437268	2.17439005	0.966232742	949.2778931	947.3900146
4.tif	37.29948936	0.822392106	5.032437765	0.929762206	1635.908691	1630.770142
						
Lee:						
Name	PSNR	SSI	SMPI	SSIM	MEAN_in	MEAN_out
0.tif	42.2768391	0.710656464	0.712393736	0.952308856	768.9508667	768.9533081
1.tif	40.38535955	0.714789271	0.715704489	0.93171881	994.5228271	994.5215454
2.tif	30.04881944	0.772177041	0.778966218	0.752842745	2816.346924	2816.355713
3.tif	43.0293077	0.749131441	0.74926873	0.962201883	949.2778931	949.2780762
4.tif	36.49870298	0.819004118	0.81980339	0.921042246	1635.908691	1635.907715
						
LeeSigma:						
Name	PSNR	SSI	SMPI	SSIM	MEAN_in	MEAN_out
0.tif	41.15573976	0.706406355	1.775652415	0.927383069	768.9508667	767.432251
1.tif	39.48654413	0.726948082	1.89393449	0.901580204	994.5228271	996.1239624
2.tif	28.67544046	0.750797272	8.862017101	0.622061956	2816.346924	2805.497803
3.tif	40.82882144	0.728175998	1.115376235	0.923971786	949.2778931	949.8087769
4.tif	34.56157946	0.839630961	2.20550154	0.847612851	1635.908691	1637.532837
![image](https://user-images.githubusercontent.com/32778343/118484113-7b50b000-b734-11eb-938a-a8740618383e.png)
