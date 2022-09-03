from torch import nn
from data_create import *

pdist = nn.PairwiseDistance(p=2)#欧式距离
cos_sim = nn.CosineSimilarity(dim=-1 , eps=1e-7)#余弦相似度

data= CasesDataset(root='./data/CasesDataset')

def ctest_bert(data):#无监督bert
    recall_num = 0
    recall = 0
    t_num =0
    print(len(data))
    for g in data:
        x1 = g.case_vector1
        x2 = g.case_vector2
        x3 = g.case_vector3
        x4 = g.case_vector4
        if g.y== 1:
            recall_num += 1
            if cos_sim(x1,x2)>=cos_sim(x3,x4):
                t_num += 1
                recall += 1
        if g.y==0:
            if cos_sim(x1,x2)<cos_sim(x3,x4):
                t_num += 1

    return t_num / len(data), recall / recall_num

r = ctest_bert(data)
print(r)
print(2*(r[0]*r[1])/(r[0]+r[1]))
