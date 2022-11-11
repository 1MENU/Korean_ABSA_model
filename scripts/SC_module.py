from util.module_utils import *
from base_data import *

from SC_dataset import *
from SC_model import *

task_name = 'SC'

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
    
    for batchIdx, (input_ids, token_type_ids, input_mask, label, e1_mask, e2_mask) in enumerate(data_loader):
        model.zero_grad() #model weight 초기화

        input_ids = input_ids.to(device) #move param_buffers to gpu
        token_type_ids = token_type_ids.to(device)
        input_mask = input_mask.to(device)
        label = label.long().to(device)
        
        e1_mask = e1_mask.to(device)
        e2_mask = e2_mask.to(device)

        output = model(input_ids, token_type_ids, input_mask, e1_mask, e2_mask) #shape: 
        
        loss = lf(output, label)
        all_loss.append(loss)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler != False : scheduler.step()

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
    min_loss = min(min_loss, avg_loss)

    y_pred = np.argmax(y_pred, axis=1)

    f1_w = f1_score(y_true, y_pred, average = 'weighted')

    print("f1 = ", f1_w,", loss = ", avg_loss)
    if wandb_on:
        wandb.log({"Train_f1": f1_w, "Train_loss": avg_loss})

    return min_loss



def eval_model(model, data_loader, lf, device, wandb_on):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []

    for batchIdx, (input_ids, token_type_ids, input_mask, label, e1_mask, e2_mask) in enumerate(data_loader):
        with torch.no_grad():
            model.zero_grad() #model weight 초기화

            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            input_mask = input_mask.to(device)
            label = label.long().to(device)
            
            e1_mask = e1_mask.to(device)
            e2_mask = e2_mask.to(device)

            output = model(input_ids, token_type_ids, input_mask, e1_mask, e2_mask) #shape: 
            
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

    f1_w = f1_score(y_true, y_pred, average = 'weighted')
    
    print('eval_f1 = ', f1_w, " eval_loss = ", avg_loss)
    if wandb_on:
        wandb.log({"eval_f1": f1_w, "eval_loss" : avg_loss})

    return f1_w, avg_loss

def SC_inference_model(model, data_loader, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    for batchIdx, (input_ids, token_type_ids, input_mask, label, e1_mask, e2_mask) in enumerate(data_loader):
        with torch.no_grad():
            model.zero_grad() #model weight 초기화

            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            input_mask = input_mask.to(device)
            label = label.long().to(device)
            
            e1_mask = e1_mask.to(device)
            e2_mask = e2_mask.to(device)

            output = model(input_ids, token_type_ids, input_mask, e1_mask, e2_mask)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    y_pred_softmax = softmax(y_pred)
    
    y_pred = np.argmax(y_pred, axis=1)

    f1_w = f1_score(y_true, y_pred, average = 'weighted')

    print('test_f1 = ', f1_w)
    
    return y_pred_softmax