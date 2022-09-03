import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'


from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from snippets import *
from torch import nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-8


class case_bert(torch.nn.Module):
    def __init__(self, detach=False):
        super(case_bert, self).__init__()#继承父类也就是module的初始化方法

        self.detach = detach
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
            # ReLU(),
            # Lin(64, 15),
        )


    def forward(self, case_s, case_t):
        pair_case_sim = self.mlp_distance(case_s - case_t)  # (64*15)
        ged = torch.cat((pair_case_sim, pair_case_sim, pair_case_sim, pair_case_sim), 1)
        return 0, 0, ged

    def loss_ged(self, y, ged_0, ged_1):
        ged = ged_0-ged_1
        y_p = self.mlp_ged(ged)
        y_p = torch.atan(y_p) / torch.pi + 1 / 2#考虑下留不留
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
                if y_p[i] >= 0.5:
                    num += 1
                    recall_num +=1
            if y[i]==0 and y_p[i] < 0.5:
                num += 1

        return num,recall_num/recall
