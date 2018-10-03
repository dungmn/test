import pickle, glob,sys
from collections import defaultdict
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.metrics import accuracy_score
from clustering import getFeaturesFromTrainOrTest

def getLabel(path):
    with open(path) as f:
        data = list(map(lambda x: x.split('/')[1], f.readlines()))
    return data

def solution(clf, iddb):
    data = getLabel('db/db{}/train.txt'.format(iddb))
    lb_dict = defaultdict(list)
    for i in range(len(data)):
        lb_dict[clf.labels_[i]].append(data[i])
    for i in lb_dict:
        lb_dict[i] = max(set(lb_dict[i]),key=lb_dict[i].count)
    trainY = getLabel('db/db{}/test.txt'.format(iddb))
    testX = getFeaturesFromTrainOrTest('db/db{}/test.txt'.format(iddb))
    idY = clf.predict(testX)
    testY = ['']*len(idY)

    for i in range(len(idY)):
        testY[i] = lb_dict[idY[i]]

    print(accuracy_score(trainY,testY))

# def mysolution(clf, typefeat, iddb):
#     lb_dict = {}
#
#     cato, list_feat = getPathFiles(typefeat)
#     center_dataset = []
#
#     for db in list_feat:
#         tmp = []
#         for p in db:
#             tmp.append(pickle.load(open(p,'rb'))[0])
#         center_dataset.append(sum(tmp)/len(tmp))
#
#     tree = KDTree(center_dataset)
#     dist, l_ind= tree.query(clf.cluster_centers_,k = 1)
#
#     for i in range(len(l_ind)):
#         lb_dict[i] = cato[l_ind[i][0]]
#     trainY = getLabel('db/db{}/test.txt'.format(iddb))
#     testX = getFeaturesFromTrainOrTest('db/db{}/test.txt'.format(iddb))
#     idY = clf.predict(testX)
#     testY = ['']*len(idY)
#
#     for i in range(len(idY)):
#         testY[i] = lb_dict[idY[i]]
#     # print(trainY[0])
#     print(accuracy_score(trainY,testY))


def evakmeans(iddb):
    path_model = 'exp/db/db{}/kmeans.pkl'.format(iddb)
    clf = pickle.load(open(path_model,'rb'))
    solution(clf,iddb)
if __name__ == '__main__':
    evakmeans(int(sys.argv[1]))
