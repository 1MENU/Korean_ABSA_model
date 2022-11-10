from util.module_utils import *
from base_data import *

class FCLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class SC_model(nn.Module):
    def __init__(self, pretrained_model):
        super(SC_model, self).__init__()

        config = AutoConfig.from_pretrained(pretrained_model)
        
        self.model = AutoModel.from_pretrained(pretrained_model)

        self.model.resize_token_embeddings(config.vocab_size + len(special_tokens_dict['additional_special_tokens']))
        
        self.cls_fc_layer = FCLayer(256, 256, 0.1)
        self.entity_fc_layer1 = FCLayer(config.hidden_size, config.hidden_size, 0.1)  # config.hidden_size
        
        self.pooler = Attention_pooler(config)
        
        self.label_classifier = FCLayer(
            256 + config.hidden_size,
            3,
            dropout_rate = 0.0,
            use_activation=False,
        )

    def forward(self, input_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states = True
        )
        
        pooled_cls = self.pooler(outputs)
        
        pooled_cls = self.cls_fc_layer(pooled_cls)

        second_cls = self.entity_average(outputs['last_hidden_state'], e2_mask)
        
        second_cls = self.entity_fc_layer1(second_cls)
        
        output = torch.cat([pooled_cls, second_cls], dim=-1)
        
        logits = self.label_classifier(output)

        return logits
    
    
    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
    
 
class Attention_pooler(nn.Module):
    def __init__(self, config):
        super(Attention_pooler, self).__init__()
        self.num_classes = 2
        self.embed_dim = config.hidden_size
        self.num_layers = 12
        self.fc_hid_dim = 256
        self.device = torch.device('cuda')
        
        self.dropout = nn.Dropout(0.1)
        
        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.embed_dim))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.embed_dim, self.fc_hid_dim))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)
        
        
    def forward(self, x):
        
        # num_layer 만큼의 hidden_states들을 stack해서 attention 적용
        
        hidden_states = x['hidden_states']

        hidden_states = torch.stack([hidden_states[-layer_i][:, 0].squeeze() for layer_i in range(1, self.num_layers+1)], dim=-1)
        
        hidden_states = hidden_states.view(-1, self.num_layers, self.embed_dim)
        
        out = self.attention(hidden_states)
        
        return out
    
    def attention(self, h):
        
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        
        return v