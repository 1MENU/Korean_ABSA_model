from util.utils import *

homePth = getParentPath(os.getcwd())
datasetPth = homePth + '/dataset/'

saveDirPth_str = homePth + '/materials/model_save/'
predPth = homePth + '/materials/pred/'
submissionPth = homePth + '/materials/submission/'


from sklearn.model_selection import StratifiedKFold

def stratified_KFold(file_list, n_splits, which_k, label_name):

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True)

    data = pd.DataFrame()
    
    for data_file in file_list:
        data = pd.concat([data, pd.read_csv(os.path.join(datasetPth, data_file), sep="\t")])

    features = data.iloc[:,:]

    label = data[label_name]

    n_iter = 0

    for train_idx, test_idx in skf.split(features, label):
        n_iter += 1

        # label_train = label.iloc[train_idx]
        # label_test = label.iloc[test_idx]

        features_train = features.iloc[train_idx]
        features_test = features.iloc[test_idx]

        if n_iter == which_k:
            print(f'------------------ {n_splits}-Fold 중 {n_iter}번째 ------------------')

            # print(features_test)

            return features_train, features_test


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):

    fname = "../dataset/" + fname

    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_TSNE(out, y_true):
    tsne_np = TSNE(n_components = 2).fit_transform(out)

    tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])

    tsne_df['target'] = y_true

    tsne_df_0 = tsne_df[tsne_df['target'] == 0]
    tsne_df_1 = tsne_df[tsne_df['target'] == 1]

    area = 2**2

    plt.scatter(tsne_df_0['component 0'], tsne_df_0['component 1'], s = area, color = 'pink', label = 'setosa')
    plt.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], s = area, color = 'purple', label = 'versicolor')

    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.legend()

    plt.savefig('boston.png')

def softmax(x):
    
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x