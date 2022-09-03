from data_create import *

data= CasesDataset(root='./data/CasesDataset')

def get_max(edge_index,edge_num):
    max = edge_index[1][edge_num - 1]
    for i in range(edge_num):
        if max < edge_index[1][i]:
            max = edge_index[1][i]
    return max

def get_graph_all_num(data):
    edge_num1 = len(data.edge_index1[0])
    node_num1 = get_max(data.edge_index1,edge_num1)
    edge_num2 = len(data.edge_index2[0])
    node_num2 = get_max(data.edge_index2,edge_num2)
    edge_num3 = len(data.edge_index3[0])
    node_num3 = get_max(data.edge_index3,edge_num3)
    edge_num4 = len(data.edge_index4[0])
    node_num4 = get_max(data.edge_index4,edge_num4)
    return edge_num1+node_num1+edge_num2+node_num2,edge_num3+node_num3+edge_num4+node_num4


def get_Ged(data):

    recall_num = 0
    recall = 0
    t_num = 0
    t = 0
    print(len(data))
    for g in data:
        x1, x2 = get_graph_all_num(g)
        r1 = g.y_0[2]*2/x1
        r2 = g.y_1[2]*2/x2
        if g.y == 1:
            recall_num += 1
            if r1 <= r2:#当相等时默认为true
                t_num += 1
                recall += 1
        if r1 > r2 and g.y == 0:
            t_num += 1
        # if t==4:
        #     break
        # t +=1

    return t_num / len(data), recall / recall_num


r = get_Ged(data)
print(r)
print(2*(r[0]*r[1])/(r[0]+r[1]))