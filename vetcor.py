
from transformers import BertTokenizer, BertModel
from snippets import *
'''
代码未经测试，运行可能会有错误
'''
model_config="./bert/bert_config.json"
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", config=model_config)


def predict(texts):
    """句子列表转换为句向量
    """
    batch_token_ids, batch_segment_ids = [], []
    for text in texts:
        sen_code = tokenizer.encode_plus(text, max_length=512, padding=True, truncation=True)
        batch_token_ids.append(sen_code['input_ids'])
        batch_segment_ids.append(sen_code['token_type_ids'])

    batch_token_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in batch_token_ids],
                                                      batch_first=True)  # 对齐张量的长度，batch_first指batch维度放在第一维
    batch_segment_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in batch_segment_ids], batch_first=True)
    model.eval()  # 为了保证预测结果一致
    # 进行编码
    with torch.no_grad():  # 不进行反向传播
        outputs = model(batch_token_ids, token_type_ids=batch_segment_ids)
        return outputs



if __name__ == '__main__':#最长的图有27个三元组，下标为826,case_text存的是张量，图节点和边存的是列表
    cases,triples = load_data(data_json)

    x=len(triples)#提取出图结构

    all_node_feature,all_edge_index, all_edge_attr=[],[],[]
    new_edge = []
    for i in range(x):

        node_feature, edge_index, edge_attr = [], [[] for i in range(2)], []
        y= len(triples[i])
        for j in range(y):
            triples[i][j][0]=triples[i][j][0].split("_")[0]
            triples[i][j][2] = triples[i][j][2].split("_")[0]

            if triples[i][j][0] not in node_feature:
                node_feature.append(triples[i][j][0])
                edge_index[0].append(len(node_feature)-1)
            else:
                edge_index[0].append(node_feature.index(triples[i][j][0]))
            if triples[i][j][2] not in node_feature:
                node_feature.append(triples[i][j][2])
                edge_index[1].append(len(node_feature)-1)
            else:
                edge_index[1].append(node_feature.index(triples[i][j][2]))
            edge_attr.append(triples[i][j][1])
            if triples[i][j][1] not in new_edge:
                new_edge.append(triples[i][j][1])
        all_node_feature.append(node_feature)
        all_edge_index.append(edge_index)
        all_edge_attr.append(edge_attr)

    print("向量化中，所需时间较长请耐心等待……")
    new_edge_vector = predict(new_edge)#将边转化为向量,边属性一共十几种，留着单独嵌入


    all_node_feature_vector,all_node_feature_pool_vector=[],[]#将节点属性转化为向量
    j=0
    for data in all_node_feature:
        outputs = predict(data)
        all_node_feature_vector.append(outputs[0])
        all_node_feature_pool_vector.append(outputs[1])


    all_node_feature_pool_vector = sequence_padding(all_node_feature_pool_vector)#将节点填充对齐,否则无法保存,这里没有对齐非池化数据

    outputs = predict(cases)#将案例文本向量化
    np.save(data_save1_vector, outputs[0])#(2,512,768)
    np.save(data_save2_vector, outputs[1])#(2,768)

    # np.save(data_save1_node_feature, torch.cat(all_node_feature_vector,dim=1))
    np.save(data_save2_node_feature, torch.tensor(all_node_feature_pool_vector))#节点特征(案例数,最大节点数，768)
    np.save(data_save_edge_index, all_edge_index)#边索引
    np.save(data_save2_edge_feature, new_edge_vector[1])#边嵌入,这里是所有边类型的嵌入，
    np.save(data_sava_all_edge, new_edge)#边，边嵌入对应所有的边类型
    np.save(data_save_all_edge_attr, all_edge_attr)#边属性
    print(u'输出向量路径：%s.npy' % data_save1_node_feature)



