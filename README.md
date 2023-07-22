# Language Identification
## Goal
Given text by user prompt or text file 'input_text.txt' in input_data folder (which is generated automatically), language of the text data is detected. For now macro F1 score (0.98) was achieved by Bi-LSTM model.
## Model 
Bi-LSTM model along with FCN on the top of it was built for the classification task. Model configuration can be found in [model.py](model.py).

## Dataset
Dataset is loaded from [HuggingFace](https://www.huggingface.com). The chosen [dataset](https://huggingface.co/datasets/papluca/language-identification), was splitted into train, validation and test sets which include 70k, 10k, 10k texts and their corresponding labels.
## 
In order to test model you can follow the steps that are given: 
* Initially, you need to pull the project into your local machine; 
* Them, you should run the following snippet to install all required dependencies: 
  ```python
  python main.py -r requirements.txt
* Now you are all set to run the following snippet (Note: The source code can be found in [playground.py](src/playground.py).) 
  ```python
  python main.py --playground_only --experiment_number 27 --play_bis --cased --clean_stops --clean_punctuation
  
 ## What is new?
 In order to see the result, we need to have tokenizer to split sentence into the words. In order to do this, I used BIS model that can be found in my repository. play_bis parameter in the code snippet that was given above activate it. If you do not set it, model will use NLTK tokenizer.
 
 I hope you will enjoy it!
 
 ## Trained model
 These following data can be downloaded from corresponding links, since they exceeds size limitation of GitHub: https://drive.google.com/drive/folders/1qS6Hb_eZdWiwc9NMSoEJc5jJztpksxKE?usp=sharing
 model_structure.pickle: you need to put it into the corresponding experiment directory
 model checkpoint: you need to put it into the checkpoints directory in the corresponding experiment path
 
 ***Regards,***

***Mahammad Namazov***
