from CD_dataset import *
from CD_model import *
from CD_module import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', required = True)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}')

model_name = args.model     # "last4-1"


test_label_file_list = ["../dataset/test_labeled.jsonl"]  # test data
test_df = jsonlload(test_label_file_list)

dataset_test = CD_dataset(test_df)   # test dataset object

InferenceLoader = DataLoader(dataset_test, batch_size = args.batch_size)

pretrained_model = "monologg/koelectra-base-v3-discriminator"   # pretrained model

mymodel = RoBertaBaseClassifier(pretrained_model)  # test model

mymodel.load_state_dict(torch.load(f'{saveDirPth_str}{task_name}/{model_name}.pt'))
mymodel.to(device)

lf = LabelSmoothingLoss(smoothing = 0.00)

submission_pred, loss = inference_model(mymodel, InferenceLoader, lf, device)

# l = np.where(loss > 0.8)[0]
# pd.set_option('display.max_rows', None)
# print(test_df.loc[l])

print(submission_pred)

np.save(f'{predPth}{task_name}/{model_name}', submission_pred)