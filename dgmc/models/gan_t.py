import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN
from torch.nn import Sequential as Seq, ReLU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
'''这里的GAN是图注意力网络'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
e=1e-8
batch_size =128#这里必须要与第一页的批次同步,说起来这里不换128就没有那么容易出事呢

class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = Lin(in_channels, out_channels, bias=False)#bias即是否加偏置
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.lin3 = Lin(768, out_channels, bias=False)
        self.lin4 = Lin(768, out_channels, bias=False)
        self.lin5 = Seq(
            Lin(out_channels * 3, out_channels,bias=False),
            ReLU(),
            Lin(out_channels, 1,bias=False),
        )
        self.lin6 = Seq(
            Lin(out_channels * 3, out_channels, bias=False),
            ReLU(),
            Lin(out_channels, 1, bias=False),
        )
        self.root = Lin(in_channels, out_channels)
        self.W=torch.tensor(1.0, requires_grad=True).to(device)#x_j的参数，用于自注意力
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()
        self.root.reset_parameters()
        reset(self.lin5)
        reset(self.lin6)

    def forward(self, x, edge_index, edge_attr):
        """"""
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=self.lin1(x), edge_attr =self.lin3(edge_attr), bools=True)#propagate在内部调用message,aggregate,update
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x), edge_attr =self.lin4(edge_attr), bools=False)
        return self.root(x) + out1 + out2
    def message(self, x_i, x_j, edge_attr,bools):
        if bools:
            tem_v = self.lin5(torch.cat((x_i,x_j,edge_attr), dim=1))#在传播中加入自注意力
        else:
            tem_v = self.lin6(torch.cat((x_i, x_j, edge_attr), dim=1))
        tem_v = torch.exp(tem_v)
        add = tem_v.sum()/batch_size
        add[add==0]=e
        tem_v = tem_v/add
        tem_v[tem_v > 1] = 1
        tem_v = tem_v.expand(tem_v.size(0),x_j.size(1))
        return tem_v*self.W*x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(GAN, self).__init__()

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

    def forward(self, x, edge_index, edge_attr, *args):
        """"""
        xs = [x]
        for conv, batch_norm in zip(self.convs, self.batch_norms):

            x = conv(xs[-1], edge_index, edge_attr)
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
