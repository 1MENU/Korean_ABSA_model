from Team1.team1.scripts.CD_module import *

##monologg/KoBERT##

parser = argparse.ArgumentParser()

parser.add_argument('--name', default="defalut")
parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--seed' , type=int , default = 5, help='random seed (default: 5)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--wandb', type=int, default=1, help='wandb on / off')
parser.add_argument('--LS', type=float, default=0.00, help='label smoothing')
parser.add_argument('--save', type=int, default=0, help='model save')
parser.add_argument('--nsplit', type=int, default=7, help='n split K-Fold')
parser.add_argument('--kfold', type=int, default=0, help='n split K-Fold')
parser.add_argument('--pretrained', default="monologg/koelectra-base-v3-discriminator")
parser.add_argument('--optimizer', default="AdamW")

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}')

set_seed(args.seed, device) #random seed 정수로 고정.


train_file_list = ["train.jsonl"]
dev_file_list = ["dev.jsonl"]
test_label_file_list = ["testt.jsonl"]

if args.kfold == 0:
    train_df = tsv_to_df(train_file_list)
    dev_df = tsv_to_df(dev_file_list)
    test_df = tsv_to_df(test_label_file_list)
else:
    train_df, dev_df = stratified_KFold(train_file_list, args.nsplit, args.kfold, 'Answer(FALSE = 0, TRUE = 1)')   # train list, n_split, k번째 fold 사용, label name
    test_df = tsv_to_df(test_label_file_list)

dataset_train = BoolQA_dataset(train_df, args.pretrained)
dataset_dev = BoolQA_dataset(dev_df, args.pretrained)
dataset_test_label = BoolQA_dataset(test_df, args.pretrained)

mymodel = model_BoolQA(args.pretrained) #default: bi-BERT
mymodel.to(device)

optimizer = build_optimizer(mymodel, lr=args.lr, weight_decay=args.weight_decay, type = args.optimizer)

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
    

TrainLoader, EvalLoader, InferenceLoader = load_data(dataset_train, dataset_dev, dataset_test_label, batch_size = args.batch_size)


print(f'\n[len {task_name}] train : {len(dataset_train)}, dev : {len(dataset_dev)}, test : {len(dataset_test_label)}\n')

print(f'Task : {task_name}, Model : {args.pretrained}, Wandb : {"off" if args.wandb == 0 else "on"}, Device : {device}, Epochs : {args.epochs}')
print(f'Batch_size : {args.batch_size}, Label_smoothing : {args.LS}, RandSeedNum :{args.seed}')
print(f'Optimizer : {args.optimizer}, Learning_rate = {args.lr}, Weight_decay : {args.weight_decay}')
print('Stratified K-Fold : None') if args.kfold == 0 else print(f'Stratified K-Fold : {args.kfold} / {args.nsplit}\n')

print(args.name, '\n')


start = time.time()

for epoch in range(args.epochs):

    print(f'[epoch {epoch}]')

    minLoss = train_model(mymodel, TrainLoader, lf, optimizer, device, args.wandb) #1epoch마다 eval

    accuracy, loss = eval_model(mymodel, EvalLoader, lf, device, "eval", args.wandb)

    # test set도 일단 돌려보기
    test_accuracy, test_loss = eval_model(mymodel, InferenceLoader, lf, device, "test", args.wandb)

    if bestAcc < (accuracy + test_accuracy) / 2 :
        bestAcc = (accuracy + test_accuracy) / 2
        bestAcc_at = epoch
        print("! new high ! -> ", bestAcc)

        if args.save:
            torch.save(mymodel.state_dict(), f"{saveDirPth_str}/{task_name}/{args.name}.pt")

    if bestLoss > (loss + test_loss) / 2 :
        bestLoss = (loss + test_loss) / 2
        bestLoss_at = epoch
        print("! new low ! -> ", bestLoss)

    if epoch >= bestAcc_at + 5 and epoch >= bestLoss_at + 5 : break


if args.wandb:
    wandb.log({"Best_acc" : bestAcc, "Best_Loss" : bestLoss})

print('\nFinish')
print(f'bestAccuracy : {bestAcc}(Best Accuracy around epoch {bestAcc_at})')
print(f'bestLoss : {bestLoss}(Best Loss around epoch {bestLoss_at})\n')
print("time per epoch :", (time.time() - start)/args.epochs)