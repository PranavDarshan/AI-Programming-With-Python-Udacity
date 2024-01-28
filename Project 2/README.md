# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application. <br><br>
The trained dataset consists of around 8200 images of 102 different flower species. I have acheived a maximum of 86% accuracy on the test set with learning rate = 0.0002 and epochs = 10 on the vgg19 architecture. The google_collab.ipynb file consists of all the outputs of 'Image Classifier Project.ipynb' file as well as the train.py and predict.py files. I have used the functions in futils.py in both of these python files. <br>
<h3><b>I have attached the pretrained checkpoint and the dataset for training, validation and testing in this file.</b></h3><br>

<h2>Instructions to Run </h2>

<br>
1. First download the datasel from kaggle : https://www.kaggle.com/datasets/pranavdarshan/flowers-dataset-udacity-ai-with-python/data
<br><br>
2. To train the model run the train.py script along with the command line arguements - path of the dataset to be trained (compulsory) 
<br><br>
Optional Arguements : <br>
 a. gpu - use gpu for training ( 1 for true and 0 for false) <br>
 b. learning_rate - learning rate at which the model should be trained on <br>
 c. epochs - number of epochs for which the model should be trained <br>
 d. save_dir - path to save the checkpoint <br>
 e. arch - architecture to train the model, choices=['vgg19','vgg16','vgg13','alexnet','densenet121'] <br>
 f. hidden_units - number of hidden units in the first hidden layer 
<br>
<h3><B> NOTE: Please use GPU to run the train.py since it takes a very long time to run on cpu</B></h3><br>
3. If you are unable to run train.py, download the pretrained model's checkpoint.pth from kaggle : [https://www.kaggle.com/models/pranavdarshan/flowerclassifier](https://www.kaggle.com/code/pranavdarshan/flower-classifier/output) <br><br>
4. Now use the saved checkpoint to run the predict.py by giving two compulsory arguements :<br>
 a. path of the image you want to predict <br>
 b. path where the checkpoint is saved <br><br>

 <h2>Results</h2>
The results can be found in the google_collab.ipynb file. Some results are: <br>

![image](https://github.com/PranavDarshan/AI-Programming-With-Python-Udacity/assets/65911046/96f33d94-eaa7-4623-8d91-7d1193b4c1f2) <br>
![image](https://github.com/PranavDarshan/AI-Programming-With-Python-Udacity/assets/65911046/d6dec624-ec8e-435c-aad5-1fafcef959c6) <br>

![image](https://github.com/PranavDarshan/AI-Programming-With-Python-Udacity/assets/65911046/b8361904-a9df-4888-b282-b4f198ebee0f)
![image](https://github.com/PranavDarshan/AI-Programming-With-Python-Udacity/assets/65911046/50691317-e526-43ec-9aa9-f0235f7bee80)


