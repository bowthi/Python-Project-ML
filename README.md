# Python-Project-ML


Problem Description 

The project addresses the issue of growing underwater waste in oceans and seas. It offers YoloV8 Algorithm-based underwater waste detection using image datasets taken from the Roboflow Environment..In this project, Google Colab is used to run codes for train and test the images. As it takes large storage capacity to run the codes, Collob is connected to T4 GPu for  fast processing.

Working Environment :
Colaboratory, or “Colab” for short, is a product from Google Research. Colab allows anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education. More technically, Colab is a hosted Jupyter notebook service that requires no setup to use, while providing access free of charge to computing resources including GPUs

Algorthm Used :
We have used YOLOv8 Algorithm .YOLOv8 is the newest model in the YOLO algorithm series – the most well-known family of object detection and classification models in the Computer Vision (CV) field.

Solution  Steps:

1.Goto Google Colob in your Web browser and sign up with your Mail Id.

2.Login to the Colob Environment and connect to the T4 GPU.

3.Mount the Colab with gdrive using the following code

	from google.colab import drive
	drive.mount('/content/gdrive')

4.Install the neccessary libraries in the Colaboratory

	!pip install ultralytics
	
5.Import the YOLOv8 from Ultralytics

	from ultralytics import YOLO
6..Install Roboflow in the Colaboratory to import the datasets from it.

	!pip install roboflow
	
7.Import the datasets from the Roboflow using the following command with the authentication token provided while running the import command

	import roboflow
	from roboflow import Roboflow

	roboflow.login()

	rf = Roboflow()

	project = rf.workspace("object-detect-dmjpt").project("ocean_waste")
	dataset = project.version(1).download("yolov8")

8.Train the model  with the following command

	!yolo task=detect mode=train model=yolov8s.pt data=ocean_waste-1/data.yaml epochs=20 imgsz=640
<p>
The Parameters,

Task = detect,indicates the specific task which is object detection
Mode = train , indicates the task of training a YOLO object detection model.
Model = YOLOv8s.pt, indicates the model used for training.
data={dataset.location}/data.yaml, indicates the location of the dataset in the yamlfile. The data file likely contains information about the dataset, including the paths to the images and their corresponding annotations (labels).
Epochs = 60 , indicates the number of iterations to train the  model.
Imgsz = 640, indicates the input image size for the model during training.
</p>		
9.Test the model
	
	!yolo task=detect mode=predict model=/content/ocean_waste-1/train/best.pt conf=0.25 source=/content/ocean_waste-1/predict/images
<p>
The Parameters,
	
Task = detect,specifies the task is object detection.
Mode = predict ,means a pre-trained model has been used to make predictions on new data.
Model ={train_result.location},the location or path of model that was trained previously.(best.pt - best prediction pytorch file)
</p>	
The above code displays the following output
##output
<p align="center">
    <img src ="https://github.com/bowthi/Python-Project-ML/blob/main/Screenshots/Result1.jpg">
</p>

10.Import Glob to display the predicted images

	import glob
	from IPython.display import Image, display

	for image_path in glob.glob(f'/content/runs/detect/predict/*.jpg'):
		display(Image(filename=image_path, width=600))
		print("\n")
##output
<p align="center"><img src= "https://github.com/bowthi/Python-Project-ML/blob/main/Screenshots/Image1.jpg"/></p>
<p align="center"> <img src= "https://github.com/bowthi/Python-Project-ML/blob/main/Screenshots/Image2.jpg"/></p>
<p algin="center"><img src="https://github.com/bowthi/Python-Project-ML/blob/main/Screenshots/image3.jpg"/></p>
<p align ="center"><img src= "https://github.com/bowthi/Python-Project-ML/blob/main/Screenshots/image4.jpg"/></p>
 <p align="center"><img src="https://github.com/bowthi/Python-Project-ML/blob/main/Screenshots/image5.jpg"> </p> 
 <p align="center"><img src="https://github.com/bowthi/Python-Project-ML/blob/main/Screenshots/image6.jpg"></p>    