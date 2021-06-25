import jieba
from gensim.models.word2vec import Word2Vec
###### 中文分词
seg_list = jieba.cut('我来到北京清华大学', cut_all=True)
print("全模式：" + "/".join(seg_list))
seg_list = jieba.cut('我来到北京清华大学', cut_all=False)
print("精准模式：" + "/".join(seg_list))
seg_list = jieba.cut("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") # 默认是精确模式
print("精准模式: " + "/ ".join(seg_list))
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") #默认是全模式
print("搜索引擎模式: " + "/ ".join(seg_list))
seg_list = jieba.lcut("我来到北京清华大学", cut_all=False)
print(seg_list) # 精准模式

import os
root_path = ''
jieba.set_dictionary(os.path.join(root_path, 'corpus/dict.txt.big'))
seg_list = jieba.lcut("今年他不幸得了新冠肺炎住院")
print(seg_list)

jieba.load_userdict(os.path.join(root_path, 'corpus/medical_term.txt'))
seg_list = jieba.lcut("今年他不幸得了COVID-19住院")
print(seg_list)
jieba.add_word('新冠肺炎')
seg_list = jieba.lcut("今年他不幸得了新冠肺炎住院")
print(seg_list)

seg_list = jieba.lcut("王小二是一个农民")
print(seg_list)

seg_list = jieba.lcut("发展社会主义的新乡村")
print(seg_list)
jieba.add_word('新乡村')
seg_list = jieba.lcut("发展社会主义的新乡村")
print(seg_list)

seg_list = jieba.lcut("王军虎去广州了")
print(seg_list)
seg_list = jieba.cut("王军虎去广州了。", HMM=False)
print("精准模式: " + "/ ".join(seg_list))


#####词性标注
import jieba.posseg as pseg
words = pseg.cut("王超超爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))

######
import Levenshtein
text1 = u"艾伦 图灵传"
text2 = u"艾伦.图灵传"
matching = Levenshtein.distance(text1, text2)

print(matching)
text1 = u"他小时候读过《三国演义》"
text2 = u"她小时候读过《青春之歌》"
matching = Levenshtein.hamming(text1, text2)
print(matching)

text1 = u"他小时候读过《三国演义》"
text2 = u"她小时候读过《青春之歌》"
matching = Levenshtein.jaro_winkler(text1, text2)
print(matching)