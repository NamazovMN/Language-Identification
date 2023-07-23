import os
import pickle
import torch
from preprocess import Preprocess
from postprocess import PostProcess
from vocabulary import Vocabulary
from model import LangClassifier
from training import Train


class Inference:
    """
    Class is utilized for independent inference. By saying independent we mean its independence from pre-initialized
    classes. This enables us to test our model, with pre-trained model without starting training.
    """

    def __init__(self, experiment_num: int, device: str, input_dir: str, choice_prompt: bool = True,
                 load_best: bool = False, choice: str = 'f1_score', epoch_choice: int = 0):
        """
        Initializer for the class which specifies required parameters

        :param experiment_num: experiment number for loading experiment results in specific folder
        :param device: device for model performance (can be cuda or cpu)
        :param input_dir: directory where data relevant information was saved
        :param choice_prompt: boolean variable will determine whether file or user prompt will be used as an input
        :param load_best: boolean variable specifies whether model parameters will be loaded based on best performance
        :param choice: choice of the best performance (f1_score, dev_loss, dev_accuracy)
        :param epoch_choice: epoch choice for model parameter loading
        """
        self.experiment_num = experiment_num
        self.choice_prompt = choice_prompt
        self.experiment_path = os.path.join('../results', f'experiment_{experiment_num}')
        self.check_exists()
        self.hp = self.get_hp()
        self.device = device
        self.preprocess, self.vocabulary, self.process, self.id2lab = self.set_processing_environment()
        self.model = self.set_model_environment(load_best, choice, epoch_choice)
        self.lang_menu = self.set_languages()
        self.input_dir = input_dir

    @staticmethod
    def set_languages() -> dict:
        """
        Method is utilized to set dictionary as a menu for languages in the given dataset
        :return: dictionary of languages => keys are dataset labels, values are corresponding languages
        """
        lang_menu = {
            'ar': 'arabic', 'bg': 'bulgarian', 'de': 'german', 'el': 'modern greek', 'en': 'english',
            'es': 'spanish', 'fr': 'french', 'hi': 'hindi', 'it': 'italian', 'ja': 'japanese', 'nl': 'dutch',
            'pl': 'polish', 'pt': 'portuguese', 'ru': 'russian', 'sw': 'swahili', 'th': 'thai',
            'tr': 'turkish', 'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese'}
        return lang_menu

    def check_exists(self) -> None:
        """
        Method is used to check whether such experiment exists or not.
        :return: None
        """
        if not os.path.exists(self.experiment_path):
            raise NotImplementedError('No such experiment was performed! Check results folder!')

    def get_hp(self) -> dict:
        """
        Method is utilized for collecting hyperparameters which will be used for inference setup
        :return: dictionary of hyperparameters
        """
        hp_file = os.path.join(self.experiment_path, 'hyperparams.pickle')
        with open(hp_file, 'rb') as hp_data:
            hp = pickle.load(hp_data)
        return hp

    def set_processing_environment(self) -> tuple:
        """
        Method is utilized to configure input processing for inference.
        :return: tuple of specific objects which will be used for input processing
        """
        data_structure_choice = self.hp['data_structure']
        dataset_dir = os.path.join('../datasets', data_structure_choice)
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError('Most likely dataset was deleted! Check readme how to create it!')
        ds_info_file = os.path.join(dataset_dir, 'dataset_info.pickle')
        with open(ds_info_file, 'rb') as ds_data:
            ds_info = pickle.load(ds_data)
        preprocess = Preprocess(
            use_words=ds_info['use_words'],
            window_size=ds_info['window_size'],
            window_shift=ds_info['window_shift']
        )
        vocabulary = Vocabulary(source_dir=dataset_dir, use_words=ds_info['use_words'])
        process = PostProcess(
            preprocess=preprocess,
            vocabulary=vocabulary,
            max_length=ds_info['max_length']
        )
        id2lab = {idx: label for label, idx in vocabulary.labels.items()}
        return preprocess, vocabulary, process, id2lab

    def set_model_environment(self, load_best: bool = False, choice: str = 'f1_score',
                              epoch_choice: int = 0) -> LangClassifier:
        """
        Method is utilized to set model environment. In other words, it loads and returns the model according to
        the given parameters, by using the training object.
        :param load_best: boolean variable specifies whether model parameters will be loaded based on best performance
        :param choice: choice of the best performance (f1_score, dev_loss, dev_accuracy)
        :param epoch_choice: epoch choice for model parameter loading
        :return: resulting model which will be used for inference
        """
        train_env = Train(
            preprocess=self.preprocess,
            vocabulary=self.vocabulary,
            hyperparams=self.hp,
            exp_num=self.experiment_num,
            device=self.device,
            load_best=load_best,
            choice=choice,
            epoch_choice=epoch_choice
        )
        train_env.load_model_parameters()
        return train_env.model

    def infer(self, text_input: str) -> None:
        """
        Method is utilized to perform inference for the given text input
        :param text_input: string for the text input
        :return: None
        """
        clean_text = self.preprocess.process_data(text_input)
        encoded_text = self.process.encode_text(clean_text)
        input_text = torch.LongTensor(encoded_text).to(self.device)
        batch = torch.unsqueeze(input_text, dim=0)
        output = self.model(batch)
        label = torch.argmax(output, -1).tolist()
        language = self.id2lab[label[0]]
        print(f'Given text is in {self.lang_menu[language]}')

    def infer_with_file(self) -> None:
        """
        Method is utilized to perform inference from the input file which is located in the input_data directory
        :return: None
        """
        file = os.path.join(self.input_dir, 'input_text.txt')
        if not os.path.exists(file):
            raise FileNotFoundError('Make sure there is input_text.txt file in the input_data folder!')
        with open(file, 'r') as input_data:
            lines = input_data.read()
        self.infer(lines)

    def infer_process(self) -> None:
        """
        Method is utilized to perform inference according to user's choice: prompt or text file
        :return: None
        """
        if self.choice_prompt:
            self.infer_with_file()
        else:
            txt_input = input('Please provide your text: \n')
            self.infer(txt_input)
