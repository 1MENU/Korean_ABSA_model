from CD_dataset import *
from CD_model import *
from CD_module import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', required = True)
parser.add_argument('--pretrained', default="monologg/koelectra-base-v3-discriminator")
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}')

model_name = args.model     # "last4-1"


test_file_list = ["../dataset/test.jsonl"]  # test data
test_data = jsonlload(test_file_list)

dataset_train, dataset_dev, dataset_test = get_CD_dataset(test_data, test_data, test_data, args.pretrained)

InferenceLoader = DataLoader(dataset_test, batch_size = args.batch_size)

mymodel = CD_model(args.pretrained)  # test model

mymodel.load_state_dict(torch.load(f'{saveDirPth_str}{task_name}/{model_name}.pt'))
mymodel.to(device)

lf = LabelSmoothingLoss(smoothing = 0.00)

submission_pred, loss = inference_model(mymodel, InferenceLoader, lf, device) # y_pred_softmax, custom_loss, f1

# l = np.where(loss > 0.8)[0]
# pd.set_option('display.max_rows', None)
# print(test_df.loc[l])

print(submission_pred)

np.save(f'{predPth}{task_name}/{model_name}', submission_pred)