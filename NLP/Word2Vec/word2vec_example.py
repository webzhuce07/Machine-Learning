# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import jieba
import jieba.analyse

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

root_path = ''
with open(root_path.join('data/in_the_name_of_people.txt'), 'r', encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    with open(root_path.join('data/in_the_name_of_people_segment.txt'), 'w', encoding='utf-8') as f2:
        f2.write(result)
f.close()
f2.close()


from gensim.models import word2vec
sentences = word2vec.LineSentence(root_path.join('data/in_the_name_of_people_segment.txt'))
model = word2vec.Word2Vec(sentences, min_count=1, window=3, size=100)

req_count = 5
for key in model.wv.similar_by_word('沙瑞金', topn=100):
    if len(key[0]) == 3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

print(model.wv.similarity('沙瑞金', '高育良'))
print(model.wv.similarity('李达康', '王大路'))
print(model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split()))

model.save(root_path.join('data/people.model'))
model1 = word2vec.Word2Vec.load(root_path.join('data/people.model'))
print(model.wv.similarity('沙瑞金', '高育良'))
print(model.wv.similarity('李达康', '王大路'))