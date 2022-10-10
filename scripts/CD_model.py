from util.module_utils import *
from base_data import *

class SimpleClassifier(nn.Module):

    def __init__(self, config, dropout, num_label):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(config.hidden_size, num_label)

    def forward(self, x):
        # x = self.dropout(x)
        # x = self.dense(x)                   
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.output(x)
        return x


class biLSTMClassifier(nn.Module):
    
    def __init__(self, config, num_label):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.size = 100
        
        self.lstm = nn.LSTM(
            input_size = config.hidden_size, 
            hidden_size = self.size, 
            num_layers = 1, 
            bias = True, 
            batch_first = True, 
            dropout = 0.0, 
            bidirectional = True
        )
        
        self.classifier = nn.Linear(self.size * 2, 2)

    def forward(self, x):
        
        h0 = torch.zeros(2, x.size(0), self.size).to('cuda') # (BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE)
        c0 = torch.zeros(2, x.size(0), self.size).to('cuda') # hidden state와 동일
        
        out, _ = self.lstm(x, (h0, c0))
        
        # output_concat = torch.cat([out[:, 0, :], out[:, -1, :]], dim = -1)
        
        out = self.classifier(out[:, -1, :])
        
        return out


class CD_model(nn.Module):
    def __init__(self, pretrained_model):
        super(CD_model, self).__init__()

        config = AutoConfig.from_pretrained(pretrained_model)
        
        self.model = AutoModel.from_pretrained(pretrained_model)

        self.model.resize_token_embeddings(config.vocab_size + len(special_tokens_dict['additional_special_tokens']))

        # self.labels_classifier = SimpleClassifier(config, 0.1, 2)
        self.bi_lstm = biLSTMClassifier(config, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        # , output_hidden_states = True
        
        # outputs=torch.cat([outputs['hidden_states'][9][:, 0, :], outputs['hidden_states'][10][:, 0, :], outputs['hidden_states'][11][:, 0, :], outputs['hidden_states'][12][:, 0, :]], dim = -1)
        # logits = self.labels_classifier(outputs)
        
        # cls_token = outputs['last_hidden_state'][:, 0, :]     # CLS token
        # logits = self.labels_classifier(cls_token)
        
        logits = self.bi_lstm(outputs['last_hidden_state'])
        
        return logits