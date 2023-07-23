from utilities import *


class Vocabulary:
    """
    Class is utilized to generate not only vocabulary based on the dataset, but also used to encode
    features and labels.
    """

    def __init__(self, source_dir: str, use_words: bool = False):
        """
        Initializer for the class which specifies required parameters
        :param source_dir: directory where vocabulary is saved (vocabulary saving might be vital for further works)
        :param use_words: boolean to determine word-based or ngram-based encoding will be performed
        """
        self.source_dir = source_dir
        self.use_words = use_words
        self.vocabulary, self.labels = self.get_vocab()

    def get_dataset(self) -> pd.DataFrame:
        """
        Method is utilized to collect train dataset for vocabulary creation
        :return: train dataset
        """
        ds_path = os.path.join(self.source_dir, 'raw_data.pickle')
        with open(ds_path, 'rb') as raw_ds:
            datasets = pickle.load(raw_ds)
        return datasets['train']

    def get_unique(self) -> tuple:
        """
        Method is utilized to extract unique features and labels from dataset
        :return: tuple object which carries set of unique tokens (words or ngrams) and list of unique labels
        """
        train_set = self.get_dataset()
        labels = train_set['labels'].unique()
        unique_tokens = list()
        for text in train_set['tokenized']:
            unique_tokens.extend(text)
        return set(unique_tokens), labels

    def generate_vocabulary(self) -> tuple:
        """
        Method is utilized to generate vocabulary and 'label to index' dictionary.
        :return: tuple object which carries vocabulary and label dictionaries
        """
        tokens, labels = self.get_unique()
        vocabulary = {token: idx for idx, token in enumerate(tokens)}
        vocabulary['<PAD>'] = len(vocabulary)
        vocabulary['<UNK>'] = len(vocabulary)
        lab2idx = {label: idx for idx, label in enumerate(labels)}
        return vocabulary, lab2idx

    def get_vocab(self) -> tuple:
        """
        Method performs all steps as a main function. If vocabulary was already generated, it will load it.
        Otherwise, all steps will be performed.
        This idea is vital, since vocabulary loading might be vital for further uses: independent inference process,
        resuming training for further refinements
        :return: tuple object which carries vocabulary and label dictionaries
        """
        check_dir(self.source_dir)
        vocab_info_file = f"vocab_info_{'words' if self.use_words else 'ngrams'}.pickle"
        vocab_path = os.path.join(self.source_dir, vocab_info_file)
        if vocab_info_file not in os.listdir(self.source_dir):
            vocabulary, lab2idx = self.generate_vocabulary()
            vocab_info = {
                'vocabulary': vocabulary,
                'lab2id': lab2idx
            }

            with open(vocab_path, 'wb') as vocab_info_data:
                pickle.dump(vocab_info, vocab_info_data)
        with open(vocab_path, 'rb') as vocab_info_data:
            vocab_info = pickle.load(vocab_info_data)

        return vocab_info['vocabulary'], vocab_info['lab2id']

    def __getitem__(self, token: str) -> int:
        """
        Method is utilized to encode the provided token
        :param token: any feature to be extracted. In case it does not exist in vocabulary, index of '<UNK>' will be
                      returned (OOV tokens)
        :return: corresponding index for provided token in the vocabulary
        """
        return self.vocabulary[token] if token in self.vocabulary.keys() else self.vocabulary['<UNK>']

    def lab2id(self, label: str) -> int:
        """
        Method is utilized to encode labels in the dataset
        :param label: language symbol for specific data
        :return: corresponding class number for the given label
        """
        return self.labels[label]

    def __len__(self):
        """
        Method is utilized to provide length of vocabulary (will be used for embedding layer)
        :return: number of features in the vocabulary
        """
        return len(self.vocabulary)
