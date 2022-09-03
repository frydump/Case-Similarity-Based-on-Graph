from snippets import *

addr1 = './data/train.json'
addr2 = './data/test.json'
addr3 = './data/valid.json'

def load_true_case(filename):
    """
    加载数据
    """
    with open(filename, encoding='utf-8') as f:
        t=f.read()
        t=t.split("}")
        t= [x+"}" for x in t]
        for i in range(len(t)):
            if t[i].find("{")==-1:
                del t[i]
            else:
                t[i] = json.loads(t[i])
    return t
def true_case_get():
    case_triple = load_true_case(addr1)
    case_triple.extend(load_true_case(addr2))
    case_triple.extend(load_true_case(addr3))
    cases,_ = load_data(data_json)
    a,b,c,t=[],[],[],[]
    for i in case_triple:
        a.append(cases.index(i["A"]))
        b.append(cases.index(i["B"]))
        c.append(cases.index(i["C"]))
        t.append(i["label"])
    list=[[],[]]
    true_bool=[]
    for j in range(len(a)):
        list[0].extend([a[j], a[j]])  # 对于数据保证每个AB的相似度大于AC的相似度
        list[1].extend([b[j], c[j]])
        if t[j]=="B":
            true_bool.append(1)
        else:
            true_bool.append(0)

    return list,true_bool