from util.module_utils import *

from Team1.team1.scripts.CD_dataset import *
from Team1.team1.scripts.CD_model import *

entity_property_pair = [
    '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
    '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성',
    '패키지/구성품#일반', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
    '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도',
                    ]
label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}

polarity_count = 0
entity_property_count = 0

if not os.path.exists("../saved_model/category_extraction/"):
    os.makedirs("../saved_model/category_extraction/")
if not os.path.exists("../saved_model/polarity_classification/"):
    os.makedirs("../saved_model/polarity_classification/")



bestAcc = -1 #0 ~ 1
bestLoss = 10

def train_model(model, data_loader, lf, optimizer, device, wandb_on):
    model.train() #set model training mode
    min_loss = 1

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []
    
    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        model.zero_grad() #model weight 초기화

        #QAset_token
        input_ids = input_ids.to(device) #move param_buffers to gpu
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        label = label.long().to(device)

        output = model(input_ids, token_type_ids, attention_mask) #shape: 
        
        loss = lf(output,label)
        all_loss.append(loss)
        
        loss.backward() #기울기 계산
        optimizer.step() #가중치 업데이트

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

def eval_model(model, data_loader, lf, device, data, wandb_on):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []

    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            label = label.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask)

            loss = lf(output,label)
            all_loss.append(loss)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()

    y_pred = np.argmax(y_pred, axis=1)
    result = compute_metrics(y_pred, y_true)["acc"]
    
    if data == "eval":
        print('eval_acc = ', result, " eval_loss = ", avg_loss)
        if wandb_on:
            wandb.log({"eval_acc": result, "eval_loss": avg_loss})
    
    else:
        print('test_acc = ', result, " test_loss = ", avg_loss)
        if wandb_on:
            wandb.log({"test_acc": result, "test_loss": avg_loss})
    
    return result, avg_loss


def inference_model(model, data_loader, lf, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    all_loss = []

    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            label = label.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask)

            loss = lf(output,label)
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
