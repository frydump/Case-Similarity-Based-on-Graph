from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from snippets import *
from true_case_get import true_case_get
'''
由于计算GED是np完全问题，这里需要的时间很长很长，建议先测试少量数据
'''

class CasesDataset(InMemoryDataset):
    """
    生成数据集
    """

    def __init__(self, root, num_compute = 8138*2, transform=None, pre_transform=None):#生成8138对图

        self.num_compute = num_compute
        super(CasesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): #它返回一个列表，显示未处理的原始文件名列表
        return []


    @property
    def processed_file_names(self):#它返回一个包含所有处理数据的文件名的列表
        return [r'.CasesDataset.dataset']

    def download(self):
        pass

    def process(self):
        node_feature = np.load(data_save2_node_feature + '.npy')
        edge_feature = np.load(data_save2_edge_feature + '.npy')

        edge_index = np.load(data_save_edge_index + '.npy', allow_pickle=True)
        edge_index = [[[k for k in j] for j in i] for i in edge_index]  # 这里是列表类型,里面元素为int,可以转为torch,但是需要先对齐

        edge_attr = np.load(data_save_all_edge_attr + '.npy', allow_pickle=True)
        edge_dict = np.load(data_sava_all_edge + '.npy')
        edge_attr_vector = [[edge_feature[edge_dict.tolist().index(i)] for i in j] for j in edge_attr]  # [1649,11,768]
        edge_attr_vector = sequence_padding(edge_attr_vector)#这里边填充了


        case_vector =np.load(data_save2_vector+'.npy')
        list,true_bool = true_case_get()#获取专家进行判定的案例,10204对案例
        print(len(true_bool))


        data_list = []

        for l in range(0,self.num_compute,2):
            i = list[0][l]
            j = list[1][l]
            k = list[0][l+1]
            m = list[1][l+1]

            len_i1 = len(edge_index[i][0]) - 1#获取图的节点数
            len_i2 = len(edge_index[j][0]) - 1
            len_i3 = len(edge_index[k][0]) - 1
            len_i4 = len(edge_index[m][0]) - 1

            graph_len = edge_index[i][0][len_i1] if edge_index[i][0][len_i1] > edge_index[i][1][len_i1] else edge_index[i][1][len_i1]# 获取当前节点最大值
            graph_len2 = edge_index[j][0][len_i2] if edge_index[j][0][len_i2] > edge_index[j][1][len_i2] else edge_index[j][1][len_i2]
            graph_len3 = edge_index[k][0][len_i3] if edge_index[k][0][len_i3] > edge_index[k][1][len_i3] else edge_index[k][1][len_i3]  # 获取当前节点最大值
            graph_len4 = edge_index[m][0][len_i4] if edge_index[m][0][len_i4] > edge_index[m][1][len_i4] else edge_index[m][1][len_i4]
            graph_len = graph_len + 1#图节点数
            graph_len2 = graph_len2 + 1
            graph_len3 = graph_len3 + 1
            graph_len4 = graph_len4 + 1
            trim_graph(edge_index, node_feature, graph_len, i)#修剪多重图的边和节点向量
            trim_graph(edge_index, node_feature, graph_len2, j)
            trim_graph(edge_index, node_feature, graph_len3, k)
            trim_graph(edge_index, node_feature, graph_len4, m)


            y_0= get_graph_y([edge_index[i]], [edge_attr[i]], [edge_index[j]], [edge_attr[j]])[0]#结点对应和ged
            y_1= get_graph_y([edge_index[k]], [edge_attr[k]], [edge_index[m]], [edge_attr[m]])[0]
            # y_0 = 0
            # y_1 = 0
            y= true_bool[int(l/2)]#真实结果

            x1 = torch.tensor(node_feature[i], dtype=torch.float)
            x2 = torch.tensor(node_feature[j], dtype=torch.float)
            x3 = torch.tensor(node_feature[k], dtype=torch.float)
            x4 = torch.tensor(node_feature[m], dtype=torch.float)

            edge_attr_vector1 = torch.tensor(edge_attr_vector[i], dtype=torch.float)
            edge_attr_vector2 = torch.tensor(edge_attr_vector[j], dtype=torch.float)
            edge_attr_vector3 = torch.tensor(edge_attr_vector[k], dtype=torch.float)
            edge_attr_vector4 = torch.tensor(edge_attr_vector[m], dtype=torch.float)

            edge_index1 = edge_index[i]
            edge_index2 = edge_index[j]
            edge_index3 = edge_index[k]
            edge_index4 = edge_index[m]

            case_vector1= torch.tensor(case_vector[i],dtype=torch.float)
            case_vector2 = torch.tensor(case_vector[j], dtype=torch.float)
            case_vector3 = torch.tensor(case_vector[k], dtype=torch.float)
            case_vector4 = torch.tensor(case_vector[m], dtype=torch.float)

            data = Data(x1=x1,x2=x2,x3=x3,x4=x4,edge_index1=edge_index1,edge_index2=edge_index2,edge_index3=edge_index3,edge_index4=edge_index4,
                        edge_attr_vector1=edge_attr_vector1,edge_attr_vector2=edge_attr_vector2,edge_attr_vector3=edge_attr_vector3,
                        edge_attr_vector4=edge_attr_vector4,case_vector1=case_vector1.reshape(1,-1),case_vector2=case_vector2.reshape(1,-1),
                        case_vector3=case_vector3.reshape(1,-1),case_vector4=case_vector4.reshape(1,-1),y_0=y_0,
                        y_1=y_1, y=y, No_1=i, No_2=j,No_3=k, No_4=m)
            data_list.append(data)

        data, slices = self.collate(data_list)  # slices返回的是字典
        torch.save((data, slices), self.processed_paths[0])

dataset_node_InMem = CasesDataset(root='./data/CasesDataset')
# print(num/len)
# print(type(dataset_node_InMem[0].x1))
# print(dataset_node_InMem)
# print(dataset_node_InMem[0].y_0)
