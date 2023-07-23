import os
import pickle
import pandas as pd

from preprocess import Preprocess
from vocabulary import Vocabulary


class PostProcess:
    """
    Class is utilized to make data ready for dataset creation for the model. This phase was separated from
    initial preprocessing, because of vocabulary generation must be made with raw data, before this phase.
    (Several ways exist, but we prefer granular and step by step performance)
    """

    def __init__(self, preprocess: Preprocess, vocabulary: Vocabulary, max_length: int):
        """
        Initializer for the class which specifies required parameters
        :param preprocess: Preprocess object will be used for dataset structure
        :param vocabulary: Vocabulary object will be used for data encoding
        :param max_length: maximum length of sequence (will be required by model)
        """
        self.preprocess = preprocess
        self.vocabulary = vocabulary
        self.max_length = max_length

    def set_length(self, text: list) -> list:
        """
        Method is utilized to set all sequences in the length of maximum length. If sequence is longer than it,
        then it will be truncated. Otherwise, sequence will be padded to the maximum length
        :param text: list of extracted features for one text
        :return: list of features in length of maximum length
        """
        if len(text) > self.max_length:
            return text[0: self.max_length]
        else:
            padded_text = text + ['<PAD>'] * (self.max_length - len(text))
            return padded_text

    def encode_text(self, input_text: list) -> list:
        """
        Method is utilized to encode each feature according to the generated vocabulary.
        :param input_text: list of features to be encoded
        :return: list of corresponding indexes per each feature in the given list
        """
        text = self.set_length(input_text)
        return [self.vocabulary[token] for token in text]

    def save_structure(self) -> None:
        """
        Method is utilized to save dataset information, in order to make tracking possible
        :return: None
        """
        ds_info_data = f"dataset_info.pickle"
        if ds_info_data not in os.listdir(self.vocabulary.source_dir):
            dataset_info = {
                'max_length': self.max_length,
                'use_words': self.preprocess.use_words,
                'window_size': self.preprocess.window_size,
                'window_shift': self.preprocess.window_shift
            }
            dataset_path = os.path.join(self.vocabulary.source_dir, ds_info_data)
            with open(dataset_path, 'wb') as ds_data:
                pickle.dump(dataset_info, ds_data)

    def process_dataset(self, dataset: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        This specific method performs dataset processing for model.
        :param dataset: dataset in form of DataFrame (makes applying functions simple)
        :param dataset_name: name of dataset which is required to save and load data
        :return: specific dataset (train, validation, test)
        """
        self.save_structure()
        ds_file = f"{dataset_name}.pickle"
        ds_path = os.path.join(self.vocabulary.source_dir, ds_file)
        if ds_file not in os.listdir(self.vocabulary.source_dir):
            dataset['encoded'] = dataset['tokenized'].apply(self.encode_text)
            dataset['label'] = dataset['labels'].apply(self.vocabulary.lab2id)
            with open(ds_path, 'wb') as data:
                pickle.dump(dataset, data)
        with open(ds_path, 'rb') as data:
            dataset = pickle.load(data)
        return dataset
