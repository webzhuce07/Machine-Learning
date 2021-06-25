import os
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
import Levenshtein
from gensim.models.word2vec import Word2Vec
from scipy import spatial
from sklearn.metrics import confusion_matrix

#设置目录环境
root_path = ''
pd.set_option('display.width', 1000) #设置字符显示宽度
pd.set_option('display.max_columns', 1000) #设置显示最大列
pd.set_option('display.max_rows', 1000)  #设置显示最大行

#设置jieba字典
jieba.set_dictionary(os.path.join(root_path, 'corpus/dict.txt.big'))
jieba.load_userdict(os.path.join(root_path, 'corpus/medical_term.txt'))

#装载预训练模型
med_model = Word2Vec.load(os.path.join(root_path, "model/med_word2vec.model"))
index2word_set = set(med_model.wv.index2word)

#读取txt文件
#输入：file_name: 文件地址
#输出： lines： 行内容列表
def loadfile(file_name):
    file_path = os.path.join(root_path, file_name)
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f_stop:
        for line in f_stop:
            lines.append(line.replace('\n', ''))
    return lines

#读取停用字典
#功能：读取自定义的停用词词典
#输入：file_name： 自定义停用词文件地址
#输出：stop_words： 停用词列表
def load_stop_word(file_name):
    stop_word = os.path.join(root_path, file_name)
    stop_words = []
    with open(stop_word, 'r', encoding='utf-8') as f_stop:
        for word in f_stop:
            stop_words.append(word.strip())
    return stop_words

#读取语料库
#功能：读取语料库形成字典，去掉语料中的词条类型及标识，即@@及后面部分
#输入：corpus_path:语料文件地址
#输出：word_list: 词典
def get_corpus_words(corpus_path):
    word_list = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            s = line.split('@@')
            word = s[0].strip('')
            try:
                type = s[1].strip('')
            except:
                print(s)
            word_list.append(word)
    return word_list

# 中文特殊字符转英文符号
# 功能：中文特殊字符转换成英文特殊字符
# 输入：str_cn:待处理的文本
# 输出：str_en:处理后的文本
def trans_spec_char(str_cn):
    str1 = u'！＠＃％＆（）【】：；，。《》？１２３４５６７８９０〈'
    str2 = u'!@#%&()[]:;,.<>?1234567890('
    table = {ord(x) : ord(y) for x, y in zip(str1, str2)}
    str_en = str_cn.translate(table)
    return str_en

# 删除数字
# 功能：删除文本中的数字和小数点
# 输入：text：待处理的文本
# 输出：text：处理后的文本
def del_digits(text):
    text = text.replace('.', '') #用空格取代小数点
    for i in range(10):
        text = text.replace(str(i), '')
    return text

# 文本相似度比较
# 功能：输入词与词典逐个比较，找出与词典中最相近的词
# 输入：corpus：词典，word：输入词
# 输出：matched_word(匹配词),max(匹配系数)
def word_similarity(corpus, word):
    max = 0.0
    matched_word = '未找到'
    for name in corpus:
        matching = Levenshtein.jaro_winkler(name, word)
        if matching > max:
            max = matching
            matched_word = name
    return matched_word, max

# 去停用词
# 功能：去除文本中的停用词
# 输入：text：文本
# 输出：text：去掉停用词的文本
def del_stop_word(text, stop_words):
    for word in stop_words:
        text = text.replace(word, '') #去掉停用词
    return text

# 命名实体识别
# 功能：对输入文本分词的词性进行分析，提取出flag指定的词性
# 输入：text：输入文本，flag：提取词的词性，缺省为名词
# 输出：entities：命令实体
def extract_entity(text, flag='n'):
    entities = []
    segments = pseg.cut(text.strip())
    for w, f in segments:
        if f == flag:
            entities.append(w)
    if len(entities) == 0:
        entities.append('None')
    return entities

# 计算平均词向量
# 功能：计算输入句子的词向量的平均值
# 输入：sentence：句子，词与词之间用空格隔开
# 输出：feature_vec:句子的平均词向量
def avg_feature_vector(sentence):
    words = sentence.split()
    feature_vec = np.zeros((200, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, med_model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

# 统计规则判断
# 功能：计算输入句子的词向量的平均值
# 输入：filepath：规则指标，df_data：数据表格（DataFrame）, column：比较项, expression：判别表达式
# 输出：df_data: 数据表格（DataFrame）,增加了
def rule_eval(filepath, df_data, column, expression):
    rules = pd.read_csv(filepath)
    data_array = np.array(df_data.loc[:, ['ICD4', column]])
    count = 0
    for data in data_array:
        key = list(data)[0]
        val = list(data)[1]
        rule = rules[rules['ICD4'] == list(data)[0]] #读取对应疾病编码的统计规则
        if len(rule) > 0:
            rule_min = float(rule['min'].tolist()[0])
            rule_max = float(rule['max'].tolist()[0])
            rule_25 = float(rule['25%'].tolist()[0])
            rule_50 = float(rule['50%'].tolist()[0])
            rule_75 = float(rule['75%'].tolist()[0])
            rule_mean = float(rule['mean'].tolist()[0])
            rule_std = float(rule['std'].tolist()[0])
            rule_count = float(rule['count'].tolist()[0])
            if eval(expression):
                df_data.loc[count, 'reason'] += '[' + str(column) + ']'  # 满足规则，则记录下对应的判别项
                df_data.loc[count, 'predict'] = 1.0  # 置异常标志
            else:
                df_data.loc[count, 'predict'] = 0.0  # 置正常标志
        count += 1
    return df_data

stop_words = load_stop_word(os.path.join(root_path, 'corpus/stopwords.txt'))

texts = ['徐汇神经','复旦中山卫生院']
corpus = get_corpus_words(os.path.join(root_path, 'corpus/medical_term.txt'))
for text in texts:
    print('---------------------------')
    print('需识别文本：', text)
    word = trans_spec_char(text.replace(' ', ''))
    word = del_digits(word)
    word = del_stop_word(word, stop_words)
    print(word)
    result = word_similarity(corpus, word)
    print(result)
    if result[1] < 0.618:
        result = extract_entity(text, flag='@@drug')
        if len(result) == 0:
            result = extract_entity(text, flag='@@diag')

    print('识别实体', result[0])


med_item0 = '尼莫地平 长春西汀 氯化钾 奥扎格雷钠 还原型谷胱甘肽 缬沙坦 地西泮 布洛芬 头孢西丁 艾司唑仑 吲哚美辛'
med_item1 = '全天麻胶囊(片) 参麦注射液 养血清脑丸(颗粒) 山莨菪碱 倍他司汀 泮托拉唑'
s1 = avg_feature_vector(med_item0)
s2 = avg_feature_vector(med_item1)
item_simility = 1 - spatial.distance.cosine(s1, s2)
print('诊疗记录1：', med_item0)
print('诊疗记录2：', med_item1)
print('诊疗记录相似度：%.2f%%' % (100 * item_simility))


sdata = pd.read_csv('data/训练测试案例(ICD4).csv', engine="python", encoding="utf-8")
sdata['reason'] = '' #增加一列 reason
print(sdata.head())
filepath = 'data/疾病住院费用统计(ICD4).csv'
sdata = rule_eval(filepath, sdata, 'BILL_SUM', 'val>2*rule_75')
print(sdata.head())
filepath = 'data/疾病住院年龄统计(ICD4).csv'
sdata = rule_eval(filepath, sdata, 'AGE', '(val>3*rule_50)|(val<0.2*rule_50)')
print(sdata.head())
filepath = 'data/疾病住院天数统计(ICD4).csv'
sdata = rule_eval(filepath, sdata, 'DAYS_OF_STAY', 'val>2.5*rule_75')
print(sdata.head())
predict = sdata['predict']
label = sdata['label']
cnf_matrix = confusion_matrix(label, predict)
print(cnf_matrix)
accuracy = (cnf_matrix[0, 0])/(cnf_matrix[0, 0]+cnf_matrix[0, 1])
recall = (cnf_matrix[0, 0])/(cnf_matrix[0, 0]+cnf_matrix[1, 0])
print('准确率为: %.2f%%' % (100*accuracy))
print('召回率为: %.2f%%' % (100*recall))
# tn, fp, fn, tp = confusion_matrix(label, predict).ravel()
# print(tn, fp, fn, tp)
# accuracy = tp/(tp + fp)
# recall = tp/(tp + fn)
# print('准确率为: %.2f%%' % (100*accuracy))
# print('召回率为: %.2f%%' % (100*recall))
# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score
# print('准确率为: %.2f%%' % (100 * precision_score(label, predict)))
# print('召回率为: %.2f%%' % (100 * recall_score(label, predict)))
# sdata.to_csv("result2.csv")

