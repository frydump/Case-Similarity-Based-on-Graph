import numpy as np
import networkx as nx
import torch
import json
'''
本文件用于存储一些通用的函数
'''
data_json = './data/cases_test.json'
data_save1_vector='./data/cases_fulltext_all_vector'
data_save2_vector='./data/cases_fulltext_pool_vector'
data_save1_node_feature='./data/cases_node_feature_all_vector'
data_save2_node_feature='./data/cases_node_feature_pool_vector'
data_save1_edge_feature='./data/cases_edge_feature_all_vector'
data_save2_edge_feature='./data/cases_edge_feature_pool_vector'
data_save_edge_index='./data/cases_edge_index'
data_sava_all_edge='./data/case_edge_all'
data_save_all_edge_attr='./data/case_edge_attr'
data_save_true_score_0 = './data/case_true_score_value'
data_save_true_score_1 = './data/case_true_score_index'

# list_value = np.load(data_save_true_score_0 + '.npy').tolist()
# formatList = np.load(data_save_true_score_1 + '.npy').tolist()

def load_data(filename):

    D_case, D_triples = [], []
    with open(filename, encoding='utf-8') as f:
        t=f.read()
        t = json.loads(t)
        for i in range(1649):#1649是所有案例的个数
            case = t[str(i)]["case"]
            triple = t[str(i)]["triples"]
            D_case.append(case)
            D_triples.append(triple)
    return D_case,D_triples

def sequence_padding(inputs, length=None, padding=0, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        if mode == 'post':
            pad_width[0] = (0, length - len(x))
        elif mode == 'pre':
            pad_width[0] = (length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)

def match_edge(e1,e2):
    if e1['feature']==e2['feature']:
        return True
    return False

def trim_graph(edge_index , node_feature, graph_len,i):#该函数未经测试，不保证可用
    mul_edge = []  # 修剪多重图的边和节点向量
    for l in range(len(edge_index[i][0])):
        if (edge_index[i][0][l], edge_index[i][1][l]) not in mul_edge:
            mul_edge.append((edge_index[i][0][l], edge_index[i][1][l]))
        else:
            node_feature[i][graph_len] = node_feature[i][edge_index[i][1][l]]  # 复制重边对应节点，将节点指向新的节点
            edge_index[i][1][l] = graph_len  # 修改对应的边特征
            graph_len = graph_len + 1
            mul_edge.append((edge_index[i][0][l], edge_index[i][1][l]))

def get_graph_y(x1_edge_index,x1_edge_attr,x2_edge_index,x2_edge_attr):#计算来自两个相同长度列表图的ged,需要很长时间
    index = x1_edge_index+x2_edge_index
    attr = x1_edge_attr+x2_edge_attr
    lens = len(x1_edge_index)
    list=[]
    for i,ii in zip(index,attr):
        G = nx.DiGraph()
        len_i1=len(i[0])-1
        graph_len = i[0][len_i1] if i[0][len_i1]>i[1][len_i1] else i[1][len_i1]#获取当前节点最大值
        edgelist, edge_attrlist = [], []
        for j,jj in zip(zip(i[0],i[1]),ii):
            if j in edgelist:#多重图处理
                if jj not in edgelist:
                    j = (j[0],graph_len)#这里修改只是在计算ged的时候有意义，结果只有对路径有影响……
                    graph_len= graph_len+1
            edgelist.append(j)
            edge_attrlist.append(jj)
            G.add_edges_from([j],feature=jj)
        # print(edgelist)
        # print(edge_attrlist)
        list.append(G)
    ged_list=[]
    for i,j in zip(range(lens),range(lens,lens*2)):
        if i==2:
            break
        min_ged=None
        for v in nx.optimize_edit_paths(list[i], list[j], None, match_edge):
            min_ged = v
        ged_list.append(min_ged)
    return ged_list

node_num = 15#这个是对齐后的节点数量
def extract_y(y ,approximata =False):#要将多个图的y合成一个大的y
    y1=[]
    y2=[]
    len_num = 0
    t_num=[]
    for i in y:#初始y为三元组
        i_s, i_t = [], []
        t_node_num = 0
        len_num_y = len_num * node_num
        if not approximata:
            i=i[0]#获取节点对应
            for j in i:
                if j[0]!=None and j[1]!=None:
                    i_s.append(j[0])
                    i_t.append(j[1])
                t_node_num = t_node_num+1
            t_node_num = t_node_num-len(i_s)+len(i_s)*2
        else:
            i_s = i[0]
            i_t = i[1]
            t_node_num =len(i[0])
        t_num.append(t_node_num)
        i_s = [i_l + len_num_y for i_l in i_s]
        len_num = len_num+1
        y1.extend(i_s)
        y2.extend(i_t)
    return [y1,y2],t_num

def get_edge_index(edge_index):
    tem_e1 = []
    tem_e2 = []
    tem_edge_num = []
    num_graph = 0
    for e in edge_index:
        num_add = num_graph * node_num
        e_0 = [i + num_add for i in e[0]]
        e_1 = [i + num_add for i in e[1]]
        num_graph = num_graph + 1
        tem_e1.extend(e_0)
        tem_e2.extend(e_1)
        tem_edge_num.append(len(e[0]))
    return [tem_e1, tem_e2],tem_edge_num

def get_edge_attr(edge_attr, edge_num ,batch_size):
    edge_len = int(edge_attr.size(0)/batch_size)
    index=[]
    for i in range(len(edge_num)):
        index.extend(range(i*edge_len,i*edge_len+edge_num[i]))
    return edge_attr[index]

def tranform_node_edge(data_set):
    for set in data_set:
        set.x1 = torch.from_numpy(set.x1)
        set.x2 = torch.from_numpy(set.x2)
        set.edge_attr_vector1 = torch.from_numpy(set.edge_attr_vector1)
        set.edge_attr_vector2 = torch.from_numpy(set.edge_attr_vector2)


def greedy_ot(cost_matrix, return_matching=False):#对于一个二维矩阵返回根据贪心算法求得的最优M,全局没用一个大于或小于，还真是nb……
    class cost_node:
        def __init__(self, i, j, cost):
            self.i = i
            self.j = j
            self.cost = cost
            self.up = None
            self.down = None
            self.left = None
            self.right = None
            self.prev = None
            self.next = None
        def __lt__(self, other):
            return self.cost < other.cost
        def delete(self):
            if self.right:
                self.right.left = self.left
            if self.left:
                self.left.right = self.right
            if self.up:
                self.up.down = self.down
            if self.down:
                self.down.up = self.up
        def __str__(self):
            return str((self.i,self.j, self.cost))
    num_pts = len(cost_matrix)
    C_cpu = cost_matrix.detach().cpu().numpy()
    # create cost nodes for every possible matching
    cost_node_list = np.array([np.array([cost_node(i,j,C_cpu[i,j]) for j in range(num_pts)]) for i in range(num_pts)])#生成和矩阵大小相同数量的节点类
    # Add location details
    for i in range(num_pts):
        for j in range(num_pts):#默认为方阵？加上，上下左右关系
            if i > 0:
                cost_node_list[i,j].up = cost_node_list[i-1,j]
            if j > 0:
                cost_node_list[i,j].left = cost_node_list[i, j-1]
            if i+1 < num_pts:
                cost_node_list[i,j].down = cost_node_list[i+1, j]
            if j+1 < num_pts:
                cost_node_list[i,j].right = cost_node_list[i, j+1]
    # make 1D and sort
    sorted_cost_list = np.sort(cost_node_list, axis=None)
    sorted_cost_list = sorted_cost_list[::-1]#反转数组
    # make linked list of cost_nodes
    for i in range(0, num_pts**2-1):#对短成本列表加上指针
        sorted_cost_list[i].next = sorted_cost_list[i+1]
    for i in range(1, num_pts**2):
        sorted_cost_list[i].prev = sorted_cost_list[i-1]

    head_node = cost_node(None,None,None)#添加上头结点和尾节点
    head_node.next = sorted_cost_list[0]
    sorted_cost_list[0].prev = head_node
    tail_node = cost_node(None,None,None)
    tail_node.prev = sorted_cost_list[-1]
    sorted_cost_list[-1].next = tail_node

    col_ind = [-1]*num_pts
    row_ind = [-1]*num_pts
    # Start the magic
    while head_node.next != tail_node:
        max_score_node = head_node.next
        row = max_score_node.i
        col = max_score_node.j
        col_ind[row] = col
        row_ind[col] = row
        #Delete same row nodes
        #Left
        current = max_score_node
        while True:
            if current.left == None:
                break
            current = current.left
            current.prev.next, current.next.prev = current.next, current.prev
            current.delete()
        #Right
        current = max_score_node
        while True:
            if current.right == None:
                break
            current = current.right
            current.prev.next, current.next.prev = current.next, current.prev
            current.delete()
        #Up
        current = max_score_node
        while True:
            if current.up == None:
                break
            current = current.up
            current.prev.next, current.next.prev = current.next, current.prev
            current.delete()
        #down
        current = max_score_node
        while True:
            if current.down == None:
                break
            current = current.down
            current.prev.next, current.next.prev = current.next, current.prev
            current.delete()
        head_node.next = max_score_node.next
        head_node.next.prev = head_node
        max_score_node.delete()
    loss = torch.tensor([0.0]).cuda()
    for i in range(num_pts):
        loss += cost_matrix[i,col_ind[i]]
    if return_matching:
        return loss/num_pts, (col_ind, row_ind)#第一个结果按序补行，第二个结果按序补列
    else:
        return loss/num_pts
def get_degree(edge_index, lens):
    in_degree = [0]*lens
    out_degree = [0]*lens
    for i,j in zip(edge_index[0],edge_index[1]):
        out_degree[i] +=1
        in_degree[j] +=1
    return in_degree, out_degree

