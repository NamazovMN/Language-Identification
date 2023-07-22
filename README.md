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
* If you want to train the model, the following snippet will be enough to run the best configuration that we got. (Configuration can be seen in [utilities.py](utilities.py).) 
  ```python
  python main.py --train --experiment_num 2

* If you want to jump direct to inference section: (Configuration can be seen in [utilities.py](utilities.py).)
  
* with text file input (do not forget add input_text.txt into input_data folder)
    ```python
      python main.py --infer --experiment_num 1 --from_file
* with user prompt
    ```python
      python main.py --infer --experiment_num 1

  
  
