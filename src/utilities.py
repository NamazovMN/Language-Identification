import os
import argparse
import pickle
import pandas as pd

from argparse import Namespace
from datasets import load_dataset
from preprocess import Preprocess


def set_parameters() -> Namespace:
    """
    Function is used to set user-defined project parameters
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=False, action='store_true', default=False,
                        help='Activates the training session')
    parser.add_argument('--data_structure', required=False, type=int, default=1,
                        help="Helps to track dataset information. Resulting folder: e.g., 'structure_1'")
    parser.add_argument('--resume_training', required=False, action='store_true', default=False,
                        help='Starts training from chosen epoch (the last epoch if choice was not made)')
    parser.add_argument('--infer', required=False, action='store_true', default=False,
                        help='Activates inference')
    parser.add_argument('--from_file', required=False, action='store_true', default=False,
                        help='Inference from user prompt (False) or file (True)')
    parser.add_argument('--epoch_choice', required=False, type=int, default=0,
                        help='Epoch choice will be used for loading model')
    parser.add_argument('--load_best', required=False, action='store_true', default=False,
                        help="Load the best model parameters according to user's choice")
    parser.add_argument('--load_choice', required=False, type=str, default='f1_score',
                        choices=['f1_score', 'dev_loss', 'dev_accuracy'],
                        help="User's choice for the best model to load")
    parser.add_argument('--use_words', required=False, action='store_true', default=False,
                        help="Defines whether word-based (True) or n-gram (False) based encoding will be performed")
    parser.add_argument('--batch_size', required=False, type=int, default=32,
                        help="Defines batch size. Compatibility in case of loading must be preserved by user!")
    parser.add_argument('--learning_rate', required=False, type=float, default=2e-4,
                        help="Specifies learning rate")
    parser.add_argument('--weight_decay', required=False, type=float, default=1e-4,
                        help="Specifies weight decay")
    parser.add_argument('--embedding_dim', required=False, type=int, default=512,
                        help="Specifies embedding dimension")
    parser.add_argument('--hidden_dim', required=False, type=int, default=50,
                        help="Specifies hidden size of LSTM layer")
    parser.add_argument('--num_layers', required=False, type=int, default=1,
                        help="Specifies number of LSTM layers")
    parser.add_argument('--lstm_dropout', required=False, type=float, default=0.0,
                        help="Specifies dropout rate for LSTM layers. (Should be chosen 0.0 if num_layers is set to 1)")
    parser.add_argument('--max_length', required=False, type=int, default=512,
                        help="Specifies maximum length will be considered by model")
    parser.add_argument('--epochs', required=False, type=int, default=40,
                        help='Specifies number of epochs to train the model')
    parser.add_argument('--bidirectional', required=False, action='store_false', default=True,
                        help="If true Bi-LSTM, else LSTM will be used as a model")
    parser.add_argument('--experiment_num', required=False, type=int, default=1,
                        help='Defines experiment number to track them')
    parser.add_argument('--html_tags', required=False, action='store_false', default=True,
                        help="Specifies removing html tags from raw text")
    parser.add_argument('--numbers', required=False, action='store_false', default=True,
                        help="Specifies removing numbers from raw text")
    parser.add_argument('--lower', required=False, action='store_false', default=True,
                        help="Specifies transforming whole texts to lowercase")
    parser.add_argument('--punctuation', required=False, action='store_false', default=True,
                        help="Specifies removing punctuation elements from raw text")
    parser.add_argument('--window_size', required=False, type=int, default=3,
                        help="Defines window size.(i.e., n-gram range)")
    parser.add_argument('--window_shift', required=False, type=int, default=2,
                        help="Defines window shift.(i.e., how many chars will be skipped within text in processing)")

    return parser.parse_args()


def get_parameters() -> dict:
    """
    Method is utilized to transform Namespace object into dict (will be used by project)
    :return: dictionary that includes all user-defined project parameters
    """
    parameters = dict()
    params_namespace = set_parameters()
    for argument in vars(params_namespace):
        parameters[argument] = getattr(params_namespace, argument)
    return parameters


def check_dir(directory: str) -> None:
    """
    Method is utilized to check the provided path's existence
    :param directory: path to be checked
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def collect_dataset(load_path: str, data_process: Preprocess, dataset_dir: str) -> dict:
    """
    Function is utilized to collect and perform preprocessing on the raw dataset
    :param load_path: path for loading the dataset (must be datasets compatible)
    :param data_process: preprocessing object
    :param dataset_dir: directory for saving the preprocessed dataset
    :return: dictionary for all preprocessed datasets
    """
    data = load_dataset(load_path)
    datasets = dict()
    raw_data_path = os.path.join(dataset_dir, 'raw_data.pickle')
    if not os.path.exists(raw_data_path):
        for ds_type, dataset in data.items():
            current_set = pd.DataFrame(dataset)
            current_set['tokenized'] = current_set['text'].apply(data_process.process_data)
            datasets[ds_type] = current_set
        with open(raw_data_path, 'wb') as raw_ds:
            pickle.dump(datasets, raw_ds)
    with open(raw_data_path, 'rb') as raw_ds:
        datasets = pickle.load(raw_ds)
    return datasets


def save_project_parameters(directory: str, data: dict) -> None:
    """
    Function is utilized to save project parameters
    :param directory: directory for project parameters for all experiments
    :param data: project parameters dictionary for current experiment
    :return: None
    """
    file_name = os.path.join(directory, f"project_parameters_{data['experiment_num']}.pickle")
    with open(file_name, 'wb') as params_path:
        pickle.dump(data, params_path)
