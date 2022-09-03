import jieba
from collections import Counter
import difflib
from snippets import *

punctuation = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。"
def clear_sign(stri):
    for i in punctuation:
        stri = stri.replace(i, '')
    return stri.replace('\n', '').replace('\r', '')

addr1 = './data/train.json'
addr2 = './data/test.json'
addr3 = './data/valid.json'

def edit_similar(str1,str2):
    len_str1=len(str1)
    len_str2=len(str2)
    taglist=np.zeros((len_str1+1,len_str2+1))
    for a in range(len_str1):
        taglist[a][0]=a
    for a in range(len_str2):
        taglist[0][a] = a
    for i in range(1,len_str1+1):
        for j in range(1,len_str2+1):
            if(str1[i - 1] == str2[j - 1]):
                temp = 0
            else:
                temp = 1
            taglist[i][j] = min(taglist[i - 1][j - 1] + temp, taglist[i][j - 1] + 1, taglist[i - 1][j] + 1)
    return 1-taglist[len_str1][len_str2] / max(len_str1, len_str2)

def cos_sim(str1, str2):
    co_str1 = (Counter(str1))
    co_str2 = (Counter(str2))
    p_str1 = []
    p_str2 = []
    for temp in set(str1 + str2):
        p_str1.append(co_str1[temp])
        p_str2.append(co_str2[temp])
    p_str1 = np.array(p_str1)
    p_str2 = np.array(p_str2)
    return p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))

def compare(str1, str2):
    str1 = clear_sign(str1)
    str2 = clear_sign(str2)
    if str1 == str2:
        return 1.0
    diff_result=difflib.SequenceMatcher(None,str1,str2).ratio()
    #分词
    str1=jieba.lcut(str1)
    str2 = jieba.lcut(str2)
    cos_result=cos_sim(str1, str2)
    edit_reslut=edit_similar(str1,str2)
    return cos_result*0.4+edit_reslut*0.3+0.3*diff_result

def load_true_case(filename):

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

def case_get():
    case_triple = load_true_case(addr1)
    case_triple.extend(load_true_case(addr2))
    case_triple.extend(load_true_case(addr3))
    recall_num = 0
    recall =0
    t_num = 0
    t = 0
    for c in case_triple:
        r1 = compare(c["A"],c["B"])
        r2 = compare(c["A"],c["C"])
        if c["label"]=="B":
            recall_num += 1
            if r1>=r2:
                t_num +=1
                recall +=1
        if r1<r2 and c["label"]=="C":
            t_num +=1
        # if t==4:
        #     break
        # t +=1
    return t_num/len(case_triple), recall/recall_num

r = case_get()
print(r)
# x = compare(str1,str2)
