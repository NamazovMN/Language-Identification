import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
import re

nltk.download('punkt')


class Preprocess:
    """
    Class is utilized to perform the preprocessing phase.
    Initially text is cleaned then tokenization or n-gram generation is applied according to the user's choice.
    Notice that, class works based on input text, which is easily applicable by pandas DataFrames.
    """

    def __init__(self, use_words: bool = False, window_size: int = 3, window_shift: int = 2, remove_html: bool = True,
                 remove_numbers: bool = True, remove_punctuation: bool = True, lower_case: bool = True):
        """
        Initializer for the class which specifies required parameters
        :param use_words: boolean to determine word-based or ngram-based tokenization will be performed
        :param window_size: integer to determine ngram size
        :param window_shift: integer to determine skipping step in ngram-based tokenization
        :param remove_html: boolean to determine removing (True) html tags from the given text
        :param remove_numbers: boolean to determine removing (True) numbers from the given text
        :param remove_punctuation: boolean to determine removing (True) punctuation elements from the given text
        :param lower_case: boolean to lower-case (True) the given text
        """
        self.use_words = use_words
        self.window_size = window_size
        self.window_shift = window_shift
        self.remove_html = remove_html
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lower_case = lower_case

    def extract_features(self, input_text: str) -> list:
        """
        Method is used to extract resulting features from the given text (words or ngrams)
        :param input_text: raw text to be processed (string)
        :return: list of features for the given text
        """
        return word_tokenize(input_text) if self.use_words else self.create_windows(input_text)

    def create_windows(self, input_text: str) -> list:
        """
        Method is used to extract ngram features according to the given size and shift values
        :param input_text: text to be processed (string)
        :return: list of ngrams which are extracted from the given text
        """
        length = len(input_text)
        windows = list()
        for idx in range(0, length, self.window_shift):
            window = input_text[idx: idx + self.window_size]
            if len(window) == self.window_size:
                windows.append(window)
        return windows

    def process_data(self, input_text: str) -> list:
        """
        Method is utilized to perform each step of preprocessing, sequentially.
        Initially, cleaning is performed (e.g., html, number and punctuation removal).
        Resulting text is sent to tokenization which considers word-based or ngram-based tokenization.
        :param input_text: raw text to be processed (string)
        :return: list of tokens as a result of preprocessing
        """
        result_text = input_text
        if self.remove_html:
            html_regex = re.compile(r'<[^>]+>')
            result_text = html_regex.sub('', result_text)
        if self.remove_numbers:
            num_regex = re.compile(r'[0-9]')
            result_text = num_regex.sub('', result_text)
        if self.remove_punctuation:
            result_text = ''.join([each_token for each_token in result_text if each_token not in punctuation])
        if self.lower_case:
            result_text = result_text.lower()
        features = self.extract_features(result_text)
        return features
