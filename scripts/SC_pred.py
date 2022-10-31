from SC_dataset import *
from SC_model import *
from SC_module import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', required = True)
parser.add_argument('--pretrained', default="kykim/electra-kor-base")
parser.add_argument('-bs', '--batch_size', type=int, default=1024)

args = parser.parse_args()

device = torch.device('cuda')

model_name = args.model     # "last4-1"

test_file_list = ["../dataset/data_new.jsonl"]  # test data
test_data = jsonlload(test_file_list)

dataset_train, dataset_dev, dataset_test = get_SC_dataset(test_data, test_data, test_data, args.pretrained, 90)

InferenceLoader = DataLoader(dataset_test, batch_size = args.batch_size)

mymodel = SC_model(args.pretrained)  # test model

mymodel.load_state_dict(torch.load(f'{saveDirPth_str}{task_name}/{model_name}.pt'))
mymodel.to(device)

lf = LabelSmoothingLoss(smoothing = 0.00)

submission_pred = SC_inference_model(mymodel, InferenceLoader, device)

# l = np.where(loss > 0.8)[0]
# pd.set_option('display.max_rows', None)
# print(test_df.loc[l])

print(submission_pred)

# np.save(f'{predPth}{task_name}/{model_name}', submission_pred)