from CD_module import *
from base_data import *

##monologg/KoBERT##

parser = argparse.ArgumentParser()

parser.add_argument('--name', default="defalut")
parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--seed' , type=int , default = 5, help='random seed (default: 5)')
parser.add_argument('--wandb', type=int, default=1, help='wandb on / off')
parser.add_argument('--LS', type=float, default=0.00, help='label smoothing')
parser.add_argument('--save', type=int, default=0, help='model save')
parser.add_argument('--nsplit', type=int, default=7, help='n split K-Fold')
parser.add_argument('--kfold', type=int, default=0, help='n split K-Fold')
parser.add_argument('--pretrained', default="xlm-roberta-base")
parser.add_argument('--optimizer', default="AdamW")

args = parser.parse_args()

device = torch.device('cuda:0')

set_seed(args.seed, device) #random seed 정수로 고정.


train_file_list = ["train.jsonl"]
dev_file_list = ["dev.jsonl"]
test_label_file_list = ["test.jsonl"]

if args.kfold == 0:
    train_data = jsonlload(train_file_list)
    dev_data = jsonlload(dev_file_list)
    test_data = jsonlload(test_label_file_list)
else:
    train_data, dev_data = stratified_KFold(train_file_list, args.nsplit, args.kfold, 'Answer(FALSE = 0, TRUE = 1)')   # train list, n_split, k번째 fold 사용, label name
    test_data = jsonlload(test_label_file_list)


tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

train_CD_data, train_SC_data = CD_dataset(train_data, tokenizer, 256)
TrainLoader = DataLoader(train_CD_data, batch_size = args.batch_size)


mymodel = RoBertaBaseClassifier(args.pretrained)
mymodel.to(device)


# entity_property_model_optimizer_setting
FULL_FINETUNING = True
if FULL_FINETUNING:
    entity_property_param_optimizer = list(mymodel.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    entity_property_optimizer_grouped_parameters = [
        {'params': [p for n, p in entity_property_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in entity_property_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]
else:
    entity_property_param_optimizer = list(mymodel.classifier.named_parameters())
    entity_property_optimizer_grouped_parameters = [{"params": [p for n, p in entity_property_param_optimizer]}]


optimizer = build_optimizer(entity_property_optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, type = args.optimizer)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = args.epochs * len(TrainLoader)
)

lf = LabelSmoothingLoss(smoothing = args.LS) # nn.CrossEntropyLoss()


if args.wandb:
    config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pretrained" : args.pretrained,
        "weight_decay" : args.weight_decay,
        "LS" : args.LS,
        "optimizer" : args.optimizer,
        "K-Fold" : f'{args.kfold}/{args.nsplit}'
    }

    wandb.init(entity="malmung_team1", project=task_name, name = args.name, config = config)


print(f'\n[len {task_name}] train : {len(train_data)}, dev : {len(dev_data)}, test : {len(test_data)}\n')

print(f'Task : {task_name}, Model : {args.pretrained}, Wandb : {"off" if args.wandb == 0 else "on"}, Device : {device}, Epochs : {args.epochs}')
print(f'Batch_size : {args.batch_size}, Label_smoothing : {args.LS}, RandSeedNum :{args.seed}')
print(f'Optimizer : {args.optimizer}, Learning_rate = {args.lr}, Weight_decay : {args.weight_decay}')
print('Stratified K-Fold : None') if args.kfold == 0 else print(f'Stratified K-Fold : {args.kfold} / {args.nsplit}\n')

print(args.name, '\n')


start = time.time()

for epoch in range(args.epochs):

    print(f'[epoch {epoch}]')

    minLoss = train_model(mymodel, TrainLoader, lf, optimizer, scheduler, device, args.wandb) #1epoch마다 eval

    f1 = eval_model(tokenizer, mymodel, copy.deepcopy(dev_data), device, "eval", args.wandb)

    test_f1 = eval_model(tokenizer, mymodel, copy.deepcopy(test_data), device, "test", args.wandb)

    if bestF1 < (f1 + test_f1) / 2 :
        bestF1 = (f1 + test_f1) / 2
        bestF1_at = epoch
        print("! new high ! -> ", bestF1)

        if args.save:
            torch.save(mymodel.state_dict(), f"{saveDirPth_str}/{task_name}/{args.name}.pt")

    # if bestLoss > (loss + test_loss) / 2 :
    #     bestLoss = (loss + test_loss) / 2
    #     bestLoss_at = epoch
    #     print("! new low ! -> ", bestLoss)

    # if epoch >= bestAcc_at + 5 and epoch >= bestLoss_at + 5 : break


if args.wandb:
    wandb.log({"Best_F1" : bestF1, "Best_Loss" : bestLoss})

print('\nFinish')
print(f'best F1 : {bestF1}(Best F1 around epoch {bestF1_at})')
# print(f'bestLoss : {bestLoss}(Best Loss around epoch {bestLoss_at})\n')
print("time per epoch :", (time.time() - start)/args.epochs)