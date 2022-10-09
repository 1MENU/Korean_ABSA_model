from util.module_utils import *
from base_data import *

from CD_dataset import *
from CD_model import *

task_name = 'CD'

make_directories(task_name)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

bestF1 = -1 #0 ~ 1
bestLoss = 10

from transformers import get_linear_schedule_with_warmup

def train_model(model, data_loader, lf, optimizer, scheduler, device, wandb_on):
    model.train() #set model training mode
    min_loss = 1

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []
    
    for batchIdx, (input_ids, token_type_ids, input_mask, label) in enumerate(data_loader):
        model.zero_grad() #model weight 초기화

        input_ids = input_ids.to(device) #move param_buffers to gpu
        token_type_ids = token_type_ids.to(device)
        input_mask = input_mask.to(device)
        label = label.long().to(device)

        output = model(input_ids, token_type_ids, input_mask) #shape: 
        
        loss = lf(output, label)
        all_loss.append(loss)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        # scheduler.step()

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
    min_loss = min(min_loss, avg_loss)

    y_pred = np.argmax(y_pred, axis=1)
    
    yy = y_true | y_pred

    y_true = y_true[yy == 1]
    y_pred = y_pred[yy == 1]

    f1_b = f1_score(y_true, y_pred, average = 'binary')

    print("f1 = ", f1_b,", loss = ", avg_loss)
    if wandb_on:
        wandb.log({"Train_f1": f1_b, "Train_loss": avg_loss})

    return min_loss



def eval_model(model, data_loader, lf, device, wandb_on):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []

    for batchIdx, (input_ids, token_type_ids, input_mask, label) in enumerate(data_loader):
        with torch.no_grad():
            model.zero_grad() #model weight 초기화

            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            input_mask = input_mask.to(device)
            label = label.long().to(device)

            output = model(input_ids, token_type_ids, input_mask) #shape: 
            
            loss = lf(output, label)
            all_loss.append(loss)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()

    y_pred = np.argmax(y_pred, axis=1)

    yy = y_true | y_pred

    y_true = y_true[yy == 1]
    y_pred = y_pred[yy == 1]

    f1_b = f1_score(y_true, y_pred, average = 'binary')
    
    print('eval_f1 = ', f1_b, " eval_loss = ", avg_loss)
    if wandb_on:
        wandb.log({"eval_f1": f1_b, "eval_loss" : avg_loss})

    return f1_b, avg_loss

from sklearn.metrics import confusion_matrix


def inference_model(model, data_loader, lf, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []

    for batchIdx, (input_ids, token_type_ids, input_mask, label) in enumerate(data_loader):
        with torch.no_grad():
            model.zero_grad() #model weight 초기화

            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            input_mask = input_mask.to(device)
            label = label.long().to(device)

            output = model(input_ids, token_type_ids, input_mask) #shape: 
            
            loss = lf(output, label)
            all_loss.append(loss)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()

    y_pred_softmax = softmax(y_pred)
    y_true_expand = np.expand_dims(y_true, axis=1)

    custom_loss = 1 - np.take_along_axis(y_pred_softmax, y_true_expand, axis=1)
    
    y_pred = np.argmax(y_pred, axis=1)
    
    yy = y_true | y_pred

    y_true = y_true[yy == 1]
    y_pred = y_pred[yy == 1]

    f1_b = f1_score(y_true, y_pred, average = 'binary')

    print('test_f1 = ', f1_b)

    return y_pred_softmax, custom_loss



def eval_model_(tokenizer, ce_model, data, device, wandb_on):

    y_pred = None #model prediction list
    
    ce_model.eval()

    pred_data = copy.deepcopy(data)

    for sentence in pred_data:
        form = sentence['sentence_form']
        sentence['annotation'] = []

        if type(form) != str:
            print("form type is arong: ", form)
            continue

        for pair in entity_property_pair:

            tokenized_data = tokenizer(form, pair, padding='max_length', max_length=256, truncation=True)

            input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
            attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)

            with torch.no_grad():
                ce_logits = ce_model(input_ids, attention_mask)
                if y_pred is None:
                    y_pred = ce_logits.detach().cpu().numpy()
                else:
                    y_pred = np.append(y_pred, ce_logits.detach().cpu().numpy(), axis=0)

            ce_predictions = torch.argmax(ce_logits, dim = -1)

            ce_result = label_id_to_name[ce_predictions[0]]

            if ce_result == 'True':

                sentence['annotation'].append([pair, 'positive'])   # 그냥 전부 postive로 설정. (CE 성능이 궁금한거라 상관없음)

    f1 = evaluation_f1(data, data)['category extraction result']['F1']
    
    
    print(f1)

    return f1