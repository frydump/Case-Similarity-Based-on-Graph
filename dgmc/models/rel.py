import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN
from torch.nn import Sequential as Seq, ReLU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Lin(in_channels, out_channels, bias=False)#bias即是否加偏置
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.root = Lin(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()


    def forward(self, x, edge_index, edge_attr):#边属性向量是经过对齐的，单个图对应27个边，每批次64
        """"""
        self.flow = 'source_to_target'#x经过了一个线性层，是为了方便拟合得分矩阵吗？
        out1 = self.propagate(edge_index, x=self.lin1(x))#propagate在内部调用message,aggregate,update
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x))#这两个out的内容要一致，问题在于边的输入维度和染色的输入维度不一致
        return self.root(x) + out1 + out2#这个好像是残差网络，每一层都连接了最终输出……

    def message(self, x_j):#这里的x_j是公式中邻域的节点，按照边索引来的
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class RelCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(RelCNN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelConv(in_channels, out_channels))
            self.batch_norms.append(BN(out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr, *args):#没有接受边属性，
        """"""
        xs = [x]#x是输入特征，这一步是为了不因为引用改变x吗

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(xs[-1], edge_index, edge_attr)#?这个读取-1的操作我不懂了
            x = batch_norm(F.relu(x)) if self.batch_norm else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.num_layers, self.batch_norm,
                                      self.cat, self.lin, self.dropout)
