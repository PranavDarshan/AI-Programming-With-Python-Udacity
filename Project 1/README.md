<h1>Use a Pre-trained Image Classifier to Identify Dog Breeds</h1>

<h2>Guidelines for the Project</h2>

Guidelines for the project 1 can be found below in the attached pdf. All the TODO's have already been completed in this code. Please refer the comments and docstrings for more information regarding the source file.

[Udacity Project 1 Use a Pre-trained Image Classifier to Identify Dog Breeds.pdf](https://github.com/PranavDarshan/AI-With-Python-Programming-Udacity/files/14062976/Udacity.Project.1.Use.a.Pre-trained.Image.Classifier.to.Identify.Dog.Breeds.pdf)


Referred from Udacity Website: 
https://learn.udacity.com/nanodegrees/nd089/parts/cd0184/lessons/9664b117-d773-4799-b6a3-d4640ed70218/concepts/8a893026-8790-4583-b280-264d43a0bb3a

<h2>Setup and Requirements to Run on Windows</h2>

1. Download all the source files.
2. Download Anaconda Navigator to create a working environment to run the tool by installing pytorch, transformers and torchvision libraries.
 
![image](https://github.com/PranavDarshan/AI-With-Python-Programming-Udacity/assets/65911046/d3ca14a9-b1a3-480c-aeb2-b21bbfe39cf5)


3.You can also install the dependencies by the environments.yaml file that is present in this project using annaconda. For that in your command prompt go to the current directory and type the following command. 
  ```ruby
conda env create -f environment.yaml
```
4. Also ensure that <b>run_models_batch</b> and <b>run_models_uploaded</b> files are in bat extension. If not change it to bat extension.

<h2>Running on Windows</h2>

1. You can directly run the <b>check_images.py</b> to run a single model on VS code by changing your interpreter to Annaconda environment that you have created and downloaded the libraries, base:root in my case. If you have installed the dependencies using the environment.yaml file, then the environment name present in the environment.yaml file is project1 and ensure that you use this environment to run the following program. Open a cmd terminal in VS Code and make sure you are in the correct directory.

![image](https://github.com/PranavDarshan/AI-With-Python-Programming-Udacity/assets/65911046/a6b8f7eb-49be-4611-8881-7321d55adaad)


2. While running the code you can give the input arguements
   replace "directory" for directory of pet images, "architecture" for model architecture, and "dogfile" for path of dognames as:
  ```ruby
python check_images.py --dir="directory" --arch="architecture" --dogfile="dogfile" 
```
 3. If you want to run multiple models and store the data in a text file, in the cmd terminal in VS Code, type
  ```ruby
run_models_batch.bat
```

4. To run uploaded images and store the data in a text file, in the cmd terminal in VS Code, type
  ```ruby
run_models_batch_uploaded.bat
```
<h2>Results</h2>

The results can be obtained on the :
1. vgg_pet-images.txt
2. vgg_uploaded-images.txt
3. alexnet_pet-images.txt
4. alexnet_uploaded-images.txt
5. resnet_pet-images.txt
6. resnet_uploaded-images.txt
