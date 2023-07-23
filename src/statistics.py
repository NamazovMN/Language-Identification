import pandas as pd
import os
import pickle
from vocabulary import Vocabulary
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Statistics:
    """
    Class is used to perform to generate statistics and results of the model
    """

    def __init__(self, exp_num: int, dataset_dir: str, use_words: bool = False):
        """
        Initializer for the class which specifies required parameters
        :param exp_num: experiment number for saving experiment results in specific folder
        :param dataset_dir: directory where data relevant information is kept
        :param use_words: boolean to determine word-based or ngram-based encoding will be performed
        """

        self.use_words = use_words
        self.experiment_num = exp_num
        self.dataset_dir = dataset_dir
        self.vocabulary = Vocabulary(source_dir=self.dataset_dir, use_words=self.use_words)
        self.idtolabel = {idx: label for label, idx in self.vocabulary.labels.items()}

    def id2label(self, idx: int) -> str:
        """
        Method is utilized to decode labels
        :param idx: index of corresponding language label
        :return: resulting language label
        """
        return self.idtolabel[idx]

    def get_exp_dir(self) -> str:
        experiment_dir = os.path.join('../results', f'experiment_{self.experiment_num}')
        if not os.path.exists(experiment_dir):
            raise NotImplementedError("There is not such experiment! Train the model first!")
        return experiment_dir

    def perform_init_statistics(self) -> None:
        """
        Method is utilized to generate information about dataset
        :return: None
        """
        ds_file = os.path.join(self.dataset_dir, 'raw_data.pickle')
        with open(ds_file, 'rb') as raw_ds:
            dataset = pickle.load(raw_ds)
        self.label_distribution(dataset['train'], 'Train')
        self.label_distribution(dataset['validation'], 'Validation')
        self.label_distribution(dataset['test'], 'Test')

        self.show_vocab_examples()

    def label_distribution(self, dataset: pd.DataFrame, ds_type: str) -> None:
        """
        Method is utilized to plot label distribution over given dataset
        :param dataset: provided dataset for performing statistics generation
        :param ds_type: type of dataset
        :return: None
        """
        labels = list(dataset['labels'])
        count = Counter(labels)
        count_data = {'label': list(), 'num_instances': list()}
        for language, num_data in count.items():
            count_data['label'].append(language)
            count_data['num_instances'].append(num_data)
        labels_figure = os.path.join(self.dataset_dir, f'label_distribution_{ds_type.lower()}.png')
        plt.bar(count_data['label'], count_data['num_instances'], color='maroon',
                width=0.4)
        plt.xlabel('Languages')
        plt.ylabel('Number of texts per language')
        plt.title(f'Text distribution over languages for {ds_type} Dataset')
        plt.savefig(labels_figure)
        plt.show()

    def show_vocab_examples(self) -> None:
        """
        Method is utilized to print vocabulary information to provide some insights
        :return: None
        """
        for k, (word, idx) in enumerate(self.vocabulary.vocabulary.items()):
            print(f"{word}: {idx}")
            if k == 10:
                break
        print(f"'<UNK>': {self.vocabulary['<UNK>']}")
        print(f"'<PAD>': {self.vocabulary['<PAD>']}")

    def plot_results(self) -> None:
        """
        Method is utilized to generate statistics after training results were generated
        :return: None
        """
        experiment_dir = self.get_exp_dir()
        results_file = os.path.join(experiment_dir, 'train_results.pickle')
        with open(results_file, 'rb') as results_data:
            results = pickle.load(results_data)

        self.plot_graph(list(results['epoch']), list(results['train_loss']), list(results['dev_loss']),
                        experiment_dir=experiment_dir, type_data='loss')
        self.plot_graph(list(results['epoch']), list(results['train_accuracy']), list(results['dev_accuracy']),
                        experiment_dir=experiment_dir, type_data='accuracy')
        self.generate_confusion_matrix(results, best_choice='dev_loss', experiment_dir=experiment_dir)
        self.generate_confusion_matrix(results, best_choice='dev_accuracy', experiment_dir=experiment_dir)
        self.generate_confusion_matrix(results, best_choice='f1_score', experiment_dir=experiment_dir)

    @staticmethod
    def plot_graph(epochs: list, train: list, validation: list, experiment_dir: str, type_data: str = 'accuracy'):
        """
        Method is used to plot to compare specific results according to given type_data (might be accuracy or loss)
        :param experiment_dir: directory where all experiment data is kept
        :param epochs: list of epochs from 1 to max
        :param train: specified results for training phase
        :param validation: specified results for validation phase
        :param type_data: string information specifies whether accuracy or loss will be plotted
        :return: None
        """
        plt.figure()
        plt.plot(epochs, train)
        plt.title(f'{type_data.title()} results over {len(epochs)} epochs')
        plt.plot(epochs, train, 'g', label='Train')
        plt.plot(epochs, validation, 'r', label='Validation')
        plt.xlabel('Number of epochs')
        plt.ylabel(f'{type_data.title()} results')
        plt.legend(loc=4)
        figure_path = os.path.join(experiment_dir, f'{type_data}_plot.png')
        plt.savefig(figure_path)
        plt.show()

    def generate_confusion_matrix(self, results_dict: dict, best_choice: str, experiment_dir: str) -> None:
        """
        Method is utilized to generate confusion matrix for best of one of 3 choices: dev accuracy and loss, f1 score
        :param experiment_dir: directory where all experiment data is kept
        :param results_dict: dictionary that includes all train results for this experiment
        :param best_choice: choice which confusion matrix will be generated accordingly
        :return: None
        """
        results_frame = pd.DataFrame(results_dict)
        value = results_frame[best_choice].min() if best_choice == 'dev_loss' else results_frame[best_choice].max()
        chosen_results = results_frame[results_frame[best_choice] == value]
        epoch_choice = list(chosen_results['epoch'])[0]
        output_path = os.path.join(experiment_dir, 'outputs')
        outputs_path = os.path.join(output_path, f'epoch_{epoch_choice}_dev_output.pickle')
        with open(outputs_path, 'rb') as out_data:
            predictions = pickle.load(out_data)
        outputs = pd.DataFrame(predictions)
        outputs['predicted_langs'] = outputs['predictions'].apply(self.id2label)
        outputs['target_langs'] = outputs['targets'].apply(self.id2label)
        conf_matrix = confusion_matrix(list(outputs['predicted_langs']), list(outputs['target_langs']))
        labels = [self.id2label(idx) for idx in range(len(self.idtolabel))]
        plt.figure(figsize=(10, 12), dpi=100)
        sns.set(font_scale=1.1)

        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )

        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(labels)

        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(labels)

        ax.set_title(f"Confusion Matrix for Language Identification based on {best_choice}", fontsize=14, pad=20)
        image_name = os.path.join(experiment_dir, f'confusion_matrix_{best_choice}.png')
        plt.savefig(image_name)
        plt.show()
