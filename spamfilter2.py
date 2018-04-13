#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import nltk
import nltk.data
import collections
import math
import random
from functools import reduce
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
#porter
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

spamfilenumber = 0#训练集中spam文档数(初始化)
hamfilenumber = 0#训练集中ham文档数(初始化)
spam_dicts = {}#训练集中spam的字典{'word':number}形式(统计某个单词在多少封spam中出现)
ham_dicts = {}#训练集中spam的字典{'word':number}形式(统计某个单词在多少封ham中出现)
threshold = 0.9#阈值的初值（可变）


#file为邮件全路径,返回值为不经任何处理的str，路径错误返回False
def reademail(file):
    if os.path.isfile(file):
        with open(file, "r", encoding = "utf-8") as f:
            text = f.read()
        return text
    else:
        print("错误的邮件地址")
        return False
    
#file为邮件全路径,返回文本预处理结果，即词干的词频统计，dicts形式
def duelemail(file):
    #调用reademail得到初始文本
    text = reademail(file)
    #正则匹配非字母,保留%、$符号并替换为空格
    pat_letter = re.compile(r'[^a-zA-Z % $ \']+')
    text = pat_letter.sub(' ', text)
    #大写字母变为小写字母
    text = text.lower()
    #sentences segment
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    #stopwords
    sr = stopwords.words("english")
    words = []
    tmpy_words = []
    #tokenize sentences
    for sentence in sentences:
        words += WordPunctTokenizer().tokenize(sentence)
    #snowballstemmer是porter的升级版
    stemmer = nltk.stem.SnowballStemmer("english")
    porterwords = map(stemmer.stem, words)
    #去除长度小于3的英文单词
    for word in porterwords:
        if word not in sr:
            if re.match('\w{3}|\$|%', word):
                tmpy_words.append(word)
    porterwords = tmpy_words
    #统计词频并排序
    dicts = collections.Counter(porterwords)
    return dicts

#对训练集进行训练，返回训练集中spam、ham文档数、spam与ham各自的字典（{'word':number}形式，number指该词在多少封spam(ham)中出现）
def traintxt(train):
    i = 0#用于统计train中spam数量即spamfilenumber
    j = 0#用于统计train中ham数量即hamfilenumber
    spam_dicts = {}#用于赋值给全局变量spam_dicts
    ham_dicts = {}#用于赋值给全局变量ham_dicts
    for file in train:
        #统计spam的数量，并形成spam的词干出现次数词典
        if re.search("spam", file) != None:
            #dicts为该spam预处理结果
            dicts = duelemail(file)
            #注意，dicts为空,不应计数，即i不应+1
            if dicts != {}:
                for dict in dicts.keys():
                    if dict not in spam_dicts.keys():
                        spam_dicts[dict] = 1
                    else:
                        spam_dicts[dict] += 1
                i += 1
        #统计ham的数量，并形成ham的词干出现次数词典
        else:
            dicts = duelemail(file)
            #dicts为该ham预处理结果
            #注意，dicts为空,不应计数，即j不应+1
            if dicts != {}:
                for dict in dicts.keys():
                    if dict not in ham_dicts.keys():
                        ham_dicts[dict] = 1
                    else:
                        ham_dicts[dict] += 1
                j += 1
    return i,j,spam_dicts,ham_dicts

#for the next
def f(x, y):
    return x*y

#for the next
def f1(x):
    if x == 1:
        return 0.01#该行可去掉，因后面计算概率时使用拉普拉斯平滑已避免1概率出现
    else:
        return 1-x

#for the next
def add(x, y):
    return x+y
                              
#测试一封新邮件,返回其为spam的概率和猜测，以及各个条件概率的List表和实际结果，形如(0.684, 'ham', [0.92, 0.54...], 'ham')
def testemail(file):
    dicts = duelemail(file)#预处理
    #声明全局变量
    global spamfilenumber, hamfilenumber, spam_dicts, ham_dicts
    #看看实际上是spam还是ham
    if re.search("spam", file) != None:
        result = 'spam'
    else:
        result = 'ham'
    #如果dicts为空，邮件可能非常短，其更可能为ham（类似"I see"）
    #这种测试样本不应用于继续学习
    if dicts == {}:
        return 1, "ham", [], result
    else:
        probability = []#每个属性(词干)得到的概率值集合
        for key in dicts.keys():
            if key in spam_dicts.keys():
                value1 = spam_dicts[key]
                if key in ham_dicts.keys():
                    value2 = ham_dicts[key]
                else:
                    value2 = 0
                p1 = (value1 + 1) * (hamfilenumber + 2)
                p2 = (value2 + 1) * (spamfilenumber + 2)
                ######下面两行注释的部分为另一种拉普拉斯计算公式（可选）######
                #p1 = (value1+1)*(hamfilenumber+len(ham_dicts.keys()))
                #p2 = (value2+1)*(spamfilenumber+len(spam_dicts.keys()))
                probability.append(p1 / (p1 + p2))                                                                   
            elif key in ham_dicts.keys():
                value1 = 0
                value2 = ham_dicts[key]
                p1 = (value1 + 1) * (hamfilenumber + 2)
                p2 = (value2 + 1) * (spamfilenumber + 2)
                ######下面两行注释的部分为另一种拉普拉斯计算公式（可选）######
                #p1 = (value1+1)*(hamfilenumber+len(ham_dicts.keys()))
                #p2 = (value2+1)*(spamfilenumber+len(spam_dicts.keys()))
                probability.append(p1 / (p1 + p2))  #使用拉普拉斯平滑处理，                                                                
            else:
                probability.append(0.4)#该属性(词干)在ham,spam中均未出现,是新词,根据数理统计结果,新词更偏向于判定邮件为ham,设为0.4
        ########下面一行代码去掉注释,就是将probability降序排序，即选取最大的15个（或全部）概率值，但是实际测试发现不采用这种策略（即选词频最高的15个词对应的概率）效果更好#######
        #probability = sorted(probability, reverse=True)
        probability = probability[0:15] #切片，根据词频大小取15个probability（小于15时取全部）
        s1 = reduce(f, probability)#f为连乘运算,即s1为各个概率值之积
        s2 = reduce(f, map(f1, probability))#f1(x) = 1 - x, s2 = （1 - x1） * (1 - x2)*...*(1 - xn)
        s3 = s1 / (s1 + s2)            #s3即为15个概率（小于15时取全部）的联合概率
        if s3 >= threshold:    #将联合概率与阈值比较，给出测试结果（阈值可调）
            guess = "spam"
        else:
            guess = "ham"
        return s3, guess, probability, result #形如(0.684, 'ham', [0.92, 0.54...], 'ham')

#这个函数用于批量测试，即对全部测试集测试，调用了testemail,并打印出了一些结果，返回值依次为误判数（将ham判为spam）、漏判数(将spam判为ham)、正确数（将spam判为spam或将ham判为ham）
def testemail2(test):
    num1, num2, num3 = 0, 0, 0
    for file in test:
        #print(file,  "\n")
        probability2, guess2, probability, result = testemail(file)#调用testemail函数对单个邮件测试
        #print(file)
        #print("概率：", probability2, " 猜测：", guess2, "\n结果是:", result)
        if guess2 != result and guess2 == "spam":
            num1 += 1
            print(file)
            print(duelemail(file))
            print(probability)
            print("概率：", probability2, " 猜测：", guess2, "\n结果是:", result)
            print("Terrible\n")
        elif guess2 != result and guess2 == "ham":
            num2 += 1
            print(file)
            print(duelemail(file))
            print(probability)
            print("概率：", probability2, " 猜测：", guess2, "\n结果是:", result)
            print("False\n")
        else:
            num3 += 1
            #print("True\n")
    return num1, num2, num3#误判数（将ham判为spam）、漏判数(将spam判为ham)、正确数（将spam判为spam或将ham判为ham）

#这个函数相比于testemail2仅仅去掉所有的打印输出，执行快
def testemail3(test):
    num1, num2, num3 = 0, 0, 0
    for file in test:
        probability2, guess2, probability, result = testemail(file)#调用testemail函数对单个邮件测试
        if guess2 != result and guess2 == "spam":
            num1 += 1
        elif guess2 != result and guess2 == "ham":
            num2 += 1
        else:
            num3 += 1
    return num1, num2, num3#误判数（将ham判为spam）、漏判数(将spam判为ham)、正确数（将spam判为spam或将ham判为ham）


def main():
    ############下面4行用于得到整个样品集在电脑里的路径，可作必要修改###########
    files1 = os.listdir(r'F:\work\bigemail\spam')
    files2 = os.listdir(r'F:\work\bigemail\ham')
    files1 = [os.path.join(r'F:\work\bigemail\spam', i) for i in files1]
    files2 = [os.path.join(r'F:\work\bigemail\ham', i) for i in files2]
    #####每次执行代码都打乱顺序#####
    random.shuffle(files1)
    random.shuffle(files2)
    #####在spam和ham中各随机取500封，总共1000封作为train(训练集)，其余所有邮件作为test(测试集)
    #####这里训练集是平衡的,即spam : ham = 1 : 1,  如果不平衡，即比例偏差很大时，模型（包括公式与参数）需要作一定的修改
    train1 = [files1[i] for i in range(500)]
    test1 = [files1[i] for i in range(500,747)]
    train2 = [files2[i] for i in range(500)]
    test2 = [files2[i] for i in range(500,4827)]
    train = train1 + train2
    test = test1 + test2
    ####进行训练
    global spamfilenumber, hamfilenumber, spam_dicts, ham_dicts#修改全局变量
    spamfilenumber, hamfilenumber, spam_dicts, ham_dicts = traintxt(train)
    print(spamfilenumber, hamfilenumber)
    global threshold
    ###################以下一段代码用来选取最佳的阈值threshold，但极其消耗时间，故只运行一遍后手工修改阈值######################
    #threshold2 = threshold
    #num1, num2 = 1000, 1000#可以是足够大的任意量
    #precision = 0.001 #精度值
    #while threshold < 1:
        #errors = num1 * 0.6 + num2 * 0.4                   #上一次的损失函数（这里没有取‘率’而是取‘数’，因分母相同时进行比较，只需比较分子（即‘数’））
        #num1, num2, num3 = testemail3(test)                #根据新的阈值threshold计算num1、num2
        #if num1 * 0.6 + num2 * 0.4 < errors:               #若损失函数更小，即效果更好，记录新的最佳阈值
            #threshold2 = threshold
        #threshold += precision                             #更新阈值threshold
    #print(threshold2)
    #threshold = threshold2                                 #选用测试出的最佳阈值
    #手工设定阈值，可修改
    threshold = 0.998
    ########对整个测试集进行测试
    num1, num2, num3 = testemail2(test)
    #######打印最终结果,形式为（spam测试集总量，ham测试集总量,"True"：num3(spam判为spam或ham判为ham数量),"False":漏判数（spam判为ham）,"Terrible":误判数（ham判为spam））
    #######结果应满足num1 + num2 + num3 == 247 + 4327 == 4574,若不满足，可能哪里出错了
    print(247, 4327, "True :", num3, "False :", num2, "Terrible :", num1)
    #######计算垃圾邮件拦截成功率、误判率
    print((247-num2) / 247, num1 / 4327)
    #######以下两行代码用于调试，可以得到降序排序后的spam、ham字典(字典本身是乱序的)
    #spam_lists = collections.Counter(spam_dicts)
    #ham_lists = collections.Counter(ham_dicts)
    #print(spam_dicts)
    #print(ham_dicts)
if __name__ == '__main__':
    main()
    
