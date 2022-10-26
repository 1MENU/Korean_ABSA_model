from CD_module import *
from base_data import *

parser = argparse.ArgumentParser()

parser.add_argument('--name', default="defalut")
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=9999)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--seed' , type=int , default = 1, help='random seed (default: 1)')
parser.add_argument('--wandb', type=int, default=1, help='wandb on / off')
parser.add_argument('--LS', type=float, default=0.00, help='label smoothing')
parser.add_argument('--save', type=int, default=0, help='model save')
parser.add_argument('--nsplit', type=int, default=0, help='n split K-Fold')
parser.add_argument('--kfold', type=int, default=0, help='n split K-Fold')
parser.add_argument('--pretrained', default="xlm-roberta-base")
parser.add_argument('--optimizer', default="AdamW")
parser.add_argument('--scheduler', default="None")
parser.add_argument('--early_stop', default=1)

args = parser.parse_args()

device = torch.device('cuda')

set_seed(args.seed, device) #random seed 정수로 고정.

# multiple files
train_file_list = ["train.jsonl", "dev.jsonl"]   # "1Fold_spell.jsonl", 
dev_file_list = ["1Fold.jsonl"]
test_file_list = ["test.jsonl"]

if args.kfold == 0:     # not split K-fold
    train_data = jsonlload(train_file_list)
    dev_data = jsonlload(dev_file_list)
    test_data = jsonlload(test_file_list)
else:   # split K-fold
    train_data, dev_data = custom_stratified_KFold(train_file_list, args.nsplit, args.kfold)
    test_data = jsonlload(test_file_list)


dataset_train, dataset_dev, dataset_test = get_CD_dataset(train_data, dev_data, test_data, args.pretrained, max_len = 90)

TrainLoader, DevLoader, InferenceLoader = load_data(dataset_train, dataset_dev, dataset_test, batch_size = args.batch_size)

mymodel = CD_model(args.pretrained)
mymodel.to(device)


FULL_FINETUNING = True
if FULL_FINETUNING:
    entity_property_param_optimizer = list(mymodel.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    entity_property_optimizer_grouped_parameters = [
        {'params': [p for n, p in entity_property_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in entity_property_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]


optimizer = build_optimizer(entity_property_optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, type = args.optimizer)

scheduler = build_scheduler(optimizer, name = args.scheduler)

lf = LabelSmoothingLoss(smoothing = args.LS)


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
        "scheduler" : args.scheduler,
        "K-Fold" : f'{args.kfold}/{args.nsplit}'
    }
    
    wandb_name = name_wandb(args.name, config)

    wandb.init(entity="malmung_team1", project=task_name, name = wandb_name, config = config)



print(f'\n[len {task_name}] train : {len(train_data)}, dev : {len(dev_data)}, test : {len(test_data)}\n')

print(f'Task : {task_name}, Model : {args.pretrained}, Wandb : {"off" if args.wandb == 0 else "on"}, Device : {device}, Epochs : {args.epochs}')
print(f'Batch_size : {args.batch_size}, Label_smoothing : {args.LS}, RandSeedNum :{args.seed}')
print(f'Optimizer : {args.optimizer}, Learning_rate = {args.lr}, Weight_decay : {args.weight_decay}')
print('K-Fold : None') if args.kfold == 0 else print(f'K-Fold : {args.kfold} / {args.nsplit}\n')

print(args.name, '\n')


start = time.time()

for epoch in range(args.epochs):

    print(f'[epoch {epoch}]')

    minLoss = train_model(mymodel, TrainLoader, lf, optimizer, scheduler, device, args.wandb)

    f1, loss = eval_model(mymodel, DevLoader, lf, device, args.wandb)

    if bestF1 < f1 :
        bestF1 = f1
        bestF1_at = epoch
        print("! new high ! -> ", bestF1)

        if args.save:
            torch.save(mymodel.state_dict(), f"{saveDirPth_str}{task_name}/{wandb_name if args.wandb else args.name}.pt")

    if bestLoss > loss :
        bestLoss = loss
        bestLoss_at = epoch
        print("! new low ! -> ", bestLoss)

    if epoch >= bestF1_at + 5 and epoch >= bestLoss_at + 5 and args.early_stop:
        final_epoch = bestF1_at
        break

if not args.early_stop:
    torch.save(mymodel.state_dict(), f"{saveDirPth_str}{task_name}/{wandb_name if args.wandb else args.name}.pt")

if args.wandb:
    wandb.log({"Best_F1" : bestF1, "Best_Loss" : bestLoss})

print('\nFinish')
print(f'best F1 : {bestF1} (Best F1 around epoch {bestF1_at})')
print(f'bestLoss : {bestLoss} (Best Loss around epoch {bestLoss_at})\n')
print("time per epoch :", (time.time() - start)/final_epoch)
