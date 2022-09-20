from util.model_utils import *
from base_data import *

class SimpleClassifier(nn.Module):

    def __init__(self, config, dropout, num_label):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(config.hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)                   
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class RoBertaBaseClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(RoBertaBaseClassifier, self).__init__()

        config = AutoConfig.from_pretrained(pretrained_model)
        
        self.model = AutoModel.from_pretrained(pretrained_model)

        self.model.resize_token_embeddings(config.vocab_size + 21)

        self.labels_classifier = SimpleClassifier(config, 0.1, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        return logits