from util.model_utils import *

class SimpleClassifier(nn.Module):

    def __init__(self, args, num_label):
        super().__init__()
        self.dense = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)
        self.output = nn.Linear(args.classifier_hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class RoBertaBaseClassifier(nn.Module):
    def __init__(self, args, num_label, len_tokenizer):
        super(RoBertaBaseClassifier, self).__init__()

        self.num_label = num_label
        self.xlm_roberta = XLMRobertaModel.from_pretrained(args.base_model)
        self.xlm_roberta.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(args, self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_label),
                                                labels.view(-1))

        return loss, logits