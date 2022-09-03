import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset
from snippets import *
from torch import nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

EPS = 1e-8


def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out


def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out


class DGMC(torch.nn.Module):

    def __init__(self, psi_1, psi_2, num_steps, k=-1, detach=False):#k，稀疏化参数；detach是否分离计算第一个图神经网络
        super(DGMC, self).__init__()

        self.psi_1 = psi_1#第一层GNN
        self.psi_2 = psi_2#第二层GNN
        self.full_node_num = 15
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        self.backend = 'auto'
        self.mlp = Seq(
            Lin(psi_2.out_channels, psi_2.out_channels),
            ReLU(),
            Lin(psi_2.out_channels, 1),
        )
        self.mlp_ged = Seq(
            Lin(60, 256),
            nn.Dropout(0.2),
            ReLU(),
            Lin(256, 256),
            ReLU(),
            Lin(256, 32),
            ReLU(),
            Lin(32, 1),
        )
        self.mlp_distance = Seq(
            Lin(768, 128),
            ReLU(),
            Lin(128, 15),
        )
        self.line = Seq(
            Lin(225, 128),
            ReLU(),
            Lin(128, 15),
        )
        self.in_degree_encoder = nn.Embedding(
            64, 768, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            64, 768, padding_idx=0)


    def reset_parameters(self):
        self.psi_1.reset_parameters()
        self.psi_2.reset_parameters()
        reset(self.mlp)

    def __top_k__(self, x_s, x_t):  # pragma: no cover
        r"""Memory-efficient top-k correspondence computation."""
        if LazyTensor is not None:
            x_s = x_s.unsqueeze(-2)  # [..., n_s, 1, d]
            x_t = x_t.unsqueeze(-3)  # [..., 1, n_t, d]
            x_s, x_t = LazyTensor(x_s), LazyTensor(x_t)
            S_ij = (-x_s * x_t).sum(dim=-1)

            return S_ij.argKmin(self.k, dim=2, backend=self.backend)
        else:
            x_s = x_s  # [..., n_s, d]
            x_t = x_t.transpose(-1, -2)  # [..., d, n_t]
            S_ij = x_s @ x_t

            return S_ij.topk(self.k, dim=2)[1]

    def __include_gt__(self, S_idx, s_mask, y):#将y中的真实值包含到索引张量中

        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)

        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)
        sparse_mask = gt_mask.new_zeros((s_mask.sum(), ))
        sparse_mask[row] = gt_mask
        dense_mask = sparse_mask.new_zeros((B, N_s))
        dense_mask[s_mask] = sparse_mask#去掉掩码，将稀疏掩码赋值过来

        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1
        dense_mask = dense_mask.view(B, N_s, 1) * last_entry.view(1, 1, k)

        return S_idx.masked_scatter(dense_mask, col[gt_mask])

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,case_s, x_t,#x_s是源图节点特征，edge_index_s是源图连通边的记录（2，边数），edge_attr_s是源图的边的特征矩阵
                edge_index_t, edge_attr_t, batch_t, case_t, y=None):#batch_s源图批处理向量，指示节点到图的分配，y是否用表示稀疏对应的真实匹配情况

        in_degree, out_degree = get_degree(edge_index_s, x_s.size(0))
        in_degree = torch.tensor(in_degree).to(device)
        out_degree = torch.tensor(out_degree).to(device)
        x_s = x_s +self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        in_degree, out_degree = get_degree(edge_index_t, x_t.size(0))
        in_degree = torch.tensor(in_degree).to(device)
        out_degree = torch.tensor(out_degree).to(device)
        x_t = x_t +self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)


        h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)#将源图和目标图都输入第一层GNN
        h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)
        assert h_s[0][0]==h_s[0][0] and h_t[0][0]==h_t[0][0]

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)

        assert h_s.size(0) == h_t.size(0), 'Encountered unequal batch-sizes'#判断批量大小是否相等
        (B, N_s, C_out), N_t = h_s.size(), h_t.size(1)#B为批量大小，N_s为源图节点数，C_out为图神经网络隐藏层输出维度，N_t为目标图节点数
        R_in, R_out = self.psi_2.in_channels, self.psi_2.out_channels



        S_hat = h_s @ h_t.transpose(-1, -2)  #[B, N_s, N_t, C_out]
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
        S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

        for _ in range(self.num_steps):
            S = masked_softmax(S_hat, S_mask, dim=-1)
            r_s = torch.randn((B, N_s, R_in), dtype=h_s.dtype,
                              device=h_s.device)#随机生成颜色
            r_t = S.transpose(-1, -2) @ r_s#转置相乘

            r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)#to_sparse根据掩码稀疏化
            o_s = self.psi_2(r_s, edge_index_s, edge_attr_s)
            o_t = self.psi_2(r_t, edge_index_t, edge_attr_t)#第二层GNN
            o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)

            D = o_s.view(B, N_s, 1, R_out) - o_t.view(B, 1, N_t, R_out)

            S_hat = S_hat + self.mlp(D).squeeze(-1).masked_fill(~S_mask, 0)#第三层神经网络

        S_L = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]#(960*15)

        S_0, S_L = (S_0.detach(), S_L.detach()) if self.detach else (S_0, S_L)  # 是否分离计算第一层图神经网络

        node_mask = torch.zeros(int(x_s.size(0)/15), 15,dtype = torch.long)#保持两图中最长边的数量
        for i in range(int(x_s.size(0)/15)):
            for j in range(15):
                if x_s[i*15+j][0]==0 and x_t[i * 15 + j][0] == 0:
                    break
                node_mask[i][j]=1
        t_S_L = S_L.reshape(-1,self.full_node_num,self.full_node_num)
        t_s = (S_0-S_L).reshape(t_S_L.size(0),-1)

        pair_node_score_0,pair_node_0 = t_S_L.max(-1)#行池化和列池化，（64*15）
        pair_node_score_1, pair_node_1 = t_S_L.max(-2)


        pair_node_score = ((pair_node_score_0+pair_node_score_1)/2)+self.line(t_s)
        pair_node_score[node_mask==0] = 0

        num = self.full_node_num
        batch_size = t_S_L.size(0)

        pdist = nn.PairwiseDistance(p=2)#欧式距离
        pair_node_dist = [[pdist(x_s[i * num + j],
                                 x_t[i * num + pair_node_0[i][j]]) / 2 + pdist(
            x_t[i * num + j], x_s[i * num + pair_node_1[i][j]]) / 2
                           for j in range(num)] for i in range(batch_size)]

        pair_graphy_dist = [[pdist(h_s[i][j], h_t[i][pair_node_0[i][j]])/ 2 + pdist(h_t[i][j], h_s[i][pair_node_1[i][j]]) / 2
                             for j in range(num)] for i in range(batch_size)]
        pair_node_dist = torch.tensor(pair_node_dist).to(device).detach()
        pair_graphy_dist = torch.tensor(pair_graphy_dist).to(device).detach()#这里会影响前面的参数传播

        pair_case_sim = self.mlp_distance(case_s-case_t)#(64*15)

        ged = torch.cat((pair_node_score,pair_node_dist,pair_graphy_dist,pair_case_sim),1)#(64,15*4)

        return S_0, S_L, ged


    def loss(self, S, y, reduction='mean'):#计算相应矩阵的负对数损失函数，这里并不是总的损失，而只是一个计算损失的函数

        assert reduction in ['none', 'mean', 'sum']
        if not S.is_sparse:
            val = S[y[0], y[1]]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            mask = S.__idx__[y[0]] == y[1].view(-1, 1)
            val = S.__val__[[y[0]]][mask]
        nll = -torch.log(val+EPS)
        if self.detach:
            nll.detach()
        return nll if reduction == 'none' else getattr(torch, reduction)(nll)

    def loss_ged(self, y, ged_0, ged_1):#最终损失
        mask0 = torch.randint(0,60,(y.size(0),12)).to(device)
        mask1 = torch.randint(0, 60, (y.size(0), 12)).to(device)
        ged = ged_0-ged_1
        ged_t = ged_0.scatter(1,mask0,0)-ged_1.scatter(1,mask1,0)
        pdist = nn.CosineSimilarity(dim=0, eps=1e-6)

        y_p = self.mlp_ged(ged_t)
        y_p = torch.atan(y_p) / torch.pi + 1 / 2
        loss = nn.BCELoss()#二分类交叉熵损失
        y = y.float().reshape(-1,1)

        return loss(y_p, y)

    def predict_acc(self, y, ged_0, ged_1):

        batch_size = ged_0.size(0)
        ged = ged_0 - ged_1

        y_p = self.mlp_ged(ged)
        y_p = torch.atan(y_p) / torch.pi + 1 / 2
        num = 0
        recall,recall_num = 0,0
        for i in range(batch_size):
            if y[i]==1 :
                recall +=1
            if y[i]==1 and y_p[i] >= 0.5:
                num += 1
                recall_num +=1
            if y[i]==0 and y_p[i] < 0.5:
                num += 1

        return num,recall_num/recall

    def acc(self, S, y, reduction='mean'):#计算对应预测的准确性，s是稀疏或密集矩阵，y是真实映射，r是应用于输出的特殊衰减

        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argmax(dim=-1)
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

        y[1]=torch.tensor(y[1]).to(device)
        correct = (pred == y[1]).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def hits_at_k(self, k, S, y, reduction='mean'):#计算前K的对应预测

        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            perm = S.__val__[y[0]].argsort(dim=-1, descending=True)[:, :k]#返回沿给定维度按升序排列张量值的指数,降序
            pred = torch.gather(S.__idx__[y[0]], -1, perm)

        correct = (pred == y[1].view(-1, 1)).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def __repr__(self):
        return ('{}(\n'
                '    psi_1={},\n'
                '    psi_2={},\n'
                '    num_steps={}, k={}\n)').format(self.__class__.__name__,
                                                    self.psi_1, self.psi_2,
                                                    self.num_steps, self.k)
