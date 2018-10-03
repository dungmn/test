from sklearn.cluster import AgglomerativeClustering, KMeans
import pickle, os, sys

def getFeaturesFromTrainOrTest(path):
    with open(path, 'r') as f:
        data = f.readlines()
    X = []
    for p in data:
        p = p.replace('images', 'VGG16_fc2')
        p = p.replace('jpg\n','jpg.pkl')
        a = pickle.load(open(p,'rb'))
        X.append(a[0])
    return X

class Clustering():
    def __init__(self, iddb = 0):
        self.path_exp = 'exp/db/db{}/'.format(iddb)
        self.path_train = 'db/db{}/train.txt'.format(iddb)
        self.training_features = getFeaturesFromTrainOrTest(self.path_train)
        self.mkdirExp()
    def mkdirExp(self):
        if not os.path.exists(self.path_exp):
            os.makedirs(self.path_exp)

    def clusterKMeans(self,num):
        features = self.training_features
        kmeans = KMeans(n_clusters=num, random_state=0).fit(features)
        pickle.dump(kmeans, open(self.path_exp + 'kmeans.pkl','wb'))

    def clusterAggC(self,num):
        features = self.training_features
        AggC = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                connectivity=None, linkage='ward', memory=None, n_clusters=num,
                pooling_func='deprecated').fit(features)
        pickle.dump(AggC, open(self.path_exp + 'AggC.pkl','wb'))

if __name__ == '__main__':
    clf = Clustering(int(sys.argv[2]))
    if sys.argv[1] == 'kmeans':
        clf.clusterKMeans(int(sys.argv[3]))
    elif sys.argv[1] == 'aggc':
        clf.clusterAggC(int(sys.argv[3]))
