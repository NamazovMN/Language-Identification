import torch
from torch import nn
from vocabulary import Vocabulary


class LangClassifier(nn.Module):
    """
    Class is used to build the classifier model
    """

    def __init__(self, hyperparams: dict, vocabulary: Vocabulary):
        """
        Initializer for the class which specifies required parameters
        :param hyperparams: dictionary for hyperparameters of the model
        :param vocabulary: vocabulary object will be used for embedding layer and padding index
        """
        super(LangClassifier, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=len(vocabulary),
            embedding_dim=hyperparams['emb_dim'],
            padding_idx=vocabulary['<PAD>']
        )
        self.lstm = nn.LSTM(
            input_size=hyperparams['emb_dim'],
            hidden_size=hyperparams['hid_dim'],
            bidirectional=hyperparams['bidirectional'],
            num_layers=hyperparams['num_layers'],
            batch_first=True
        )

        input_dim = hyperparams['hid_dim'] * 2 if hyperparams['bidirectional'] else hyperparams['hid_dim']
        self.dense = nn.Flatten()
        self.linear = nn.Linear(
            in_features=hyperparams['max_length'] * input_dim,
            out_features=hyperparams['num_classes']
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_data: torch.LongTensor) -> torch.Tensor:
        """
        Method is utilized to perform feedforward process
        :param input_data: input tensor for the model
        :return: output of the model
        """
        embeddings = self.embedding(input_data)
        lstm_out, (_, _) = self.lstm(embeddings)
        relu_out1 = self.relu1(lstm_out)
        dense_out = self.dense(relu_out1)
        dp_out = self.dropout(dense_out)
        relu_out = self.relu2(dp_out)
        out = self.linear(relu_out)
        return out
