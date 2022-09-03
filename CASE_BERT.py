import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

from data_create import *
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
from dgmc.models import case_bert
import argparse

'''
注意本文件代码所需的数据格式和论文方法有区别
'''
parser = argparse.ArgumentParser()#命令行解析模块


parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=55)
parser.add_argument('--test_samples', type=int, default=2000)
args = parser.parse_args()
approximate_Bool =False

data= CasesDataset(root='./data/CasesDataset2')
num_feature= data[0].x1.size(-1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = case_bert().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

drop_after_epoch = [30, 40]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.1,
                                                 last_epoch=-1)  # 动态步长


def train():
    model.train()

    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()


        data = data.to(device)

        S_0_0, S_L_0, Ged_0 = model(data.case_vector1, data.case_vector2)
        S_0_1, S_L_1, Ged_1 = model(data.case_vector3, data.case_vector4)

        loss = model.loss_ged(data.y, Ged_0, Ged_1)

        loss.backward()

        optimizer.step()
        total_loss += loss.item() * (data.x1_batch.max().item() + 1)


    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def xtest(dataset):
    model.eval()#保持结果不变
    loader = DataLoader(dataset, args.batch_size, shuffle=True,
                        follow_batch=['x1', 'x2', 'x3', 'x4'])

    correct = num_examples = acc = 0
    acc_num = 0
    recall = 0
    i = 1
    while (num_examples < args.test_samples):#这个是限制总的样例
        for data in loader:

            optimizer.zero_grad()
            data = data.to(device)

            S_0_0, S_L_0,Ged_0 = model(data.case_vector1, data.case_vector2)
            S_0_1, S_L_1,Ged_1 = model(data.case_vector3, data.case_vector4)

            tem_acc,recall_0 = model.predict_acc(data.y, Ged_0, Ged_1)
            acc += tem_acc#计算分类正确率

            recall += recall_0
            num_examples +=6
            acc_num += len(data.y)
            if num_examples >= args.test_samples:
                return correct / num_examples, acc / acc_num,recall/i
            i += 1



kf = KFold(5, shuffle=True, random_state=38)#使用5折交叉验证

all_acc_list,all_recall_list,all_f1_list = [],[],[]

k=0
for train_index, test_index in kf.split(data):
    train_set = [data[i] for i in train_index]
    test_set = [data[i] for i in test_index]

    model = case_bert().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    drop_after_epoch = [30, 40]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.1,
                                                     last_epoch=-1)  # 动态步长

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['x1','x2','x3','x4'])

    one_fold_acc_list,recall_list,F1_list = [],[],[]
    for epoch in range(1, args.epochs+1):
        loss = train()
        scheduler.step()#动态学习率

        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        if epoch % 1 == 0 or epoch > 50:
            accs,t_loss,recall = xtest(test_set)
            F1 = 2 * t_loss * recall / (t_loss + recall)
            one_fold_acc_list.append(t_loss)
            recall_list.append(recall)
            F1_list.append(F1)
            print((f'k_fold: {k}, epoch {epoch:03d}: Loss: {loss:.4f}, Average forecast true:{ t_loss:.4f}, Recall:{recall:.4f}, F1:{F1:.4f}'))

    all_acc_list.append(one_fold_acc_list)
    all_recall_list.append(recall_list)
    all_f1_list.append(F1_list)
    k += 1
print(np.mean(all_acc_list,0))
print(np.mean(all_recall_list,0))
print(np.mean(all_f1_list,0))
