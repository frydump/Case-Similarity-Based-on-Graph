import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


from data_create import *
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from dgmc.models import DGMC,GAN
import argparse



'''
图相似度计算模型
'''
parser = argparse.ArgumentParser()#命令行解析模块
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--rnd_dim', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--k', type=int, default=-1)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--test_samples', type=int, default=2000)
args = parser.parse_args()
approximate_Bool =False

data= CasesDataset(root='./data/CasesDataset')

num_feature= data[0].x1.size(-1)
# num_feature = len(data[0].x1[0])#即bert嵌入维数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
psi_1 = GAN(num_feature, args.dim, args.num_layers, batch_norm=False,
                cat=True, lin=True, dropout=0.2)  # 输入，输出，层数
psi_2 = GAN(args.rnd_dim, args.rnd_dim, args.num_layers, batch_norm=False,
            cat=True, lin=True, dropout=0.4)

model = DGMC(psi_1, psi_2, num_steps=10, k=args.k).to(device)  # num_steps共识迭代次数
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

drop_after_epoch = [50, 70]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.1,
                                                 last_epoch=-1)  # 动态步长

loss_bool = True

def train():
    model.train()
    total_loss = 0

    for data in train_loader:#训练损失递增，因为两个图神经网络隐藏层数量过多
        # y_0, node_num_0=extract_y(data.y_0, True)#这里node_num是两图真实结点数量的和
        y_0, node_num_0 = extract_y(data.y_0, approximate_Bool)
        y_1, node_num_1 = extract_y(data.y_1, approximate_Bool)

        optimizer.zero_grad()

        tem_DEI1, edge_num1 = get_edge_index(data.edge_index1)#单独处理index，直接放于datalord加载tensor类型会出错
        tem_DEI2, edge_num2 = get_edge_index(data.edge_index2)
        tem_DEI3, edge_num3 = get_edge_index(data.edge_index3)
        tem_DEI4, edge_num4 = get_edge_index(data.edge_index4)
        tem_DEI1 = torch.tensor(tem_DEI1).to(device)
        tem_DEI2 = torch.tensor(tem_DEI2).to(device)
        tem_DEI3 = torch.tensor(tem_DEI3).to(device)
        tem_DEI4 = torch.tensor(tem_DEI4).to(device)
        tem_edge_attr1 = get_edge_attr(data.edge_attr_vector1, edge_num1, args.batch_size).to(device)#单独处理边属性
        tem_edge_attr2 = get_edge_attr(data.edge_attr_vector2, edge_num2, args.batch_size).to(device)
        tem_edge_attr3 = get_edge_attr(data.edge_attr_vector3, edge_num3, args.batch_size).to(device)  # 单独处理边属性
        tem_edge_attr4 = get_edge_attr(data.edge_attr_vector4, edge_num4, args.batch_size).to(device)

        data = data.to(device)

        S_0_0, S_L_0, Ged_0 = model(data.x1, tem_DEI1, tem_edge_attr1,
                         data.x1_batch, data.case_vector1, data.x2, tem_DEI2,#x1_batch是用来划分节点所属的图的，长度等于总节点数，但是边的所属没有划分
                         tem_edge_attr2, data.x2_batch, data.case_vector2)#这里加y可以保证训练中包含正确结果，图小，感觉不用加
        S_0_1, S_L_1, Ged_1 = model(data.x3, tem_DEI3, tem_edge_attr3,
                             data.x3_batch, data.case_vector3, data.x4, tem_DEI4,  # x1_batch是用来划分节点所属的图的，长度等于总节点数，但是边的所属没有划分
                             tem_edge_attr4, data.x4_batch, data.case_vector4)  # 这里加y可以保证训练中包含正确结果，图小，感觉不用加

        if loss_bool:
            loss = model.loss(S_L_0, y_0) + model.loss(S_L_1, y_1)#如果共识迭代次数大于零则加上，否则不加
            loss = 0.1*model.loss_ged(data.y, Ged_0, Ged_1)+loss

        else:
            loss = model.loss_ged(data.y, Ged_0, Ged_1)


        loss.backward()

        optimizer.step()
        total_loss += loss.item() * (data.x1_batch.max().item() + 1)
        # flops, params = profile(model, (data.x1, tem_DEI1, tem_edge_attr1,
        #                  data.x1_batch, data.case_vector1, data.x2, tem_DEI2,#x1_batch是用来划分节点所属的图的，长度等于总节点数，但是边的所属没有划分
        #                  tem_edge_attr2, data.x2_batch, data.case_vector2,))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        # break

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
        for data in loader:#这里一个data，是一个batch
            y_0, node_num_0 = extract_y(data.y_0, approximate_Bool)  # 这里node_num是两图真实结点数量的和
            y_1, node_num_1 = extract_y(data.y_1, approximate_Bool)

            optimizer.zero_grad()

            tem_DEI1, edge_num1 = get_edge_index(data.edge_index1)  # 单独处理index，直接放于datalord加载tensor类型会出错
            tem_DEI2, edge_num2 = get_edge_index(data.edge_index2)
            tem_DEI3, edge_num3 = get_edge_index(data.edge_index3)
            tem_DEI4, edge_num4 = get_edge_index(data.edge_index4)
            tem_DEI1 = torch.tensor(tem_DEI1).to(device)
            tem_DEI2 = torch.tensor(tem_DEI2).to(device)
            tem_DEI3 = torch.tensor(tem_DEI3).to(device)
            tem_DEI4 = torch.tensor(tem_DEI4).to(device)
            tem_edge_attr1 = get_edge_attr(data.edge_attr_vector1, edge_num1, args.batch_size).to(device)  # 单独处理边属性
            tem_edge_attr2 = get_edge_attr(data.edge_attr_vector2, edge_num2, args.batch_size).to(device)
            tem_edge_attr3 = get_edge_attr(data.edge_attr_vector3, edge_num3, args.batch_size).to(device)  # 单独处理边属性
            tem_edge_attr4 = get_edge_attr(data.edge_attr_vector4, edge_num4, args.batch_size).to(device)

            data = data.to(device)

            S_0_0, S_L_0,Ged_0 = model(data.x1, tem_DEI1, tem_edge_attr1,
                                 data.x1_batch, data.case_vector1, data.x2, tem_DEI2,  # x1_batch是用来划分节点所属的图的，长度等于总节点数，但是边的所属没有划分
                                 tem_edge_attr2, data.x2_batch, data.case_vector2)  # 这里加y可以保证训练中包含正确结果，图小，感觉不用加
            S_0_1, S_L_1,Ged_1 = model(data.x3, tem_DEI3, tem_edge_attr3,
                                 data.x3_batch, data.case_vector3, data.x4, tem_DEI4,  # x1_batch是用来划分节点所属的图的，长度等于总节点数，但是边的所属没有划分
                                 tem_edge_attr4, data.x4_batch, data.case_vector4)  # 这里加y可以保证训练中包含正确结果，图小，感觉不用加
            tem_acc,recall_0 = model.predict_acc(data.y, Ged_0, Ged_1)
            acc += tem_acc#计算分类正确率
            recall += recall_0

            correct =correct + model.acc(S_L_0, y_0, reduction='sum') + model.acc(S_L_1, y_1, reduction='sum')
            num_examples += len(y_0[0]+y_1[0])
            # num_examples +=6
            acc_num += len(data.y)
            if num_examples >= args.test_samples:  # 这里是1000个结点的正确率，同时也是1000个结点对应真实情况的欧式距离
                return correct / num_examples, acc / acc_num,recall/i
            i += 1

# data, valid_set = train_test_split(data, test_size=1/6, random_state=42)#(640和128)
train_set, test_set = train_test_split(data, test_size=1/6, random_state=42)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['x1','x2','x3','x4'])

for epoch in range(1, args.epochs+1):

    loss = train()
    scheduler.step()#动态学习率

    if epoch % 1 == 0 or epoch > 50:#运行过程有几率陷入预测正确率为1的情况，不知道为什么
        # accs,t_loss,recall = xtest(valid_set)
        accs, t_loss,recall = xtest(test_set)
        if accs>0.8:
            loss_bool =False
            model.detach = True
        F1 = 2*t_loss*recall/(t_loss+recall)
        print((f'{epoch:03d}: Loss: {loss:.4f}, accs: {accs:.4f}, Average forecast true:{ t_loss:.4f}, Recall:{recall:.4f}, F1:{F1:.4f}'))
        # torch.save(model.state_dict(), './weights/Judicial_model.%s.weights' % epoch)
        # break
