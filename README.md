# Learning-from-Limited-and-Challenging-Visual-Data
 Runinng Instructions:
 1. The first clone the github repo. 
 2. from the link: https://drive.google.com/drive/folders/1iLWlsbIcUiWnLMwUIwckzrdQNFMEfsbh?usp=sharing download the dataset. and replace the data folder.
 3. For every method there is a directory.  
 4. Go to the method drectory that wants to run (using git bash cd command) 
 5. open the train_method_name.sh file select the data split (instruction given in .sh file)
 6. now in git bash type chmod +x train_method_name.sh
 7. then simply type ./train_method_name.sh to execute. (note: here method_name will be replace by the original file name)


 To evalute a method with a save model weight:
 1. download the data
 2. dowload the saved mode from: https://drive.google.com/drive/folders/1vf2K19p5EveKSsfX7ABQ8l4l7au64e9w?usp=sharing
 3. paste the the model weight in the method folder's saved_models dir
 4. comment "model_training" fuction in train_method_name.py file and run train_method_name.sh as before
 
 

