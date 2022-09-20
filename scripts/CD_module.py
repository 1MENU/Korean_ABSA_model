from util.module_utils import *
from base_data import *

from CD_dataset import *
from CD_model import *

task_name = 'CD'



polarity_count = 0
entity_property_count = 0

if not os.path.exists("../saved_model/category_extraction/"):
    os.makedirs("../saved_model/category_extraction/")
if not os.path.exists("../saved_model/polarity_classification/"):
    os.makedirs("../saved_mod   el/polarity_classification/")



bestF1 = -1 #0 ~ 1
bestLoss = 10

from transformers import get_linear_schedule_with_warmup

def train_model(model, data_loader, lf, optimizer, scheduler, device, wandb_on):
    model.train() #set model training mode
    min_loss = 1

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []
    
    for batchIdx, (input_ids, input_mask, label) in enumerate(data_loader):
        model.zero_grad() #model weight 초기화

        input_ids = input_ids.to(device) #move param_buffers to gpu
        input_mask = input_mask.to(device)
        label = label.long().to(device)

        output = model(input_ids, input_mask) #shape: 
        
        loss = lf(output, label)
        all_loss.append(loss)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # loss.backward() #기울기 계산
        # optimizer.step() #가중치 업데이트

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
    min_loss = min(min_loss, avg_loss)

    y_pred = np.argmax(y_pred, axis=1)
    accuracy = compute_metrics(y_pred, y_true)["acc"]

    print("acc = ", accuracy,", loss = ",avg_loss)
    if wandb_on:
        wandb.log({"Train_accuracy": accuracy, "Train_loss": avg_loss})

    return min_loss

def eval_model(tokenizer, ce_model, data, device, dataset_type, wandb_on):

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

            ce_predictions = torch.argmax(ce_logits, dim = -1)

            ce_result = label_id_to_name[ce_predictions[0]]

            if ce_result == 'True':

                sentence['annotation'].append([pair, 'positive'])   # 그냥 전부 postive로 설정. (CE 성능이 궁금한거라 상관없음)

    f1 = evaluation_f1(data, pred_data)['category extraction result']['F1']

    if dataset_type == "eval" :
        print('eval_f1 = ', f1, " eval_loss = ") 
        if wandb_on:
            wandb.log({"eval_f1": f1})    # "eval_loss": avg_loss
    
    elif dataset_type == "test" :
        print('test_acc = ', f1, " test_loss = ")
        if wandb_on:
            wandb.log({"test_f1": f1})

    return f1


def inference_model(model, data_loader, lf, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []

    for batchIdx, (input_ids, input_mask, label) in enumerate(data_loader):
        with torch.no_grad():
            model.zero_grad() #model weight 초기화

            input_ids = input_ids.to(device) #move param_buffers to gpu
            input_mask = input_mask.to(device)
            label = label.long().to(device)

            output = model(input_ids, input_mask) #shape: 
            
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
    result = compute_metrics(y_pred, y_true)["acc"]
    
    print('test_acc = ', result, " test_loss = ", avg_loss)
    
    return y_pred_softmax, custom_loss