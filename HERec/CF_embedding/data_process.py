#!/usr/bin/python3
# coding: utf-8
from numpy import *
import time
from texttable import Texttable
import numpy as np
import os


class CF_embedding:
    def __init__(self, movies, ratings, embeddings, k,n):
        self.movies = movies
        self.ratings = ratings
        self.embeddings = embeddings
        # 邻居个数
        self.k = k
        # 推荐个数
        self.n = n
        # 用户对电影的评分
        # 数据格式{'UserID：用户ID':[(MovieID：电影ID,Rating：用户对电影的评星)]}
        self.userDict = {}
        # 对某电影评分的用户
        # 数据格式：{'MovieID：电影ID',[UserID：用户ID]}
        # {'1',[1,2,3..],...}
        self.ItemUser = {}
        # 邻居的信息
        self.neighbors = []
        # 推荐列表
        self.recommandList = []
        self.cost_p = 0.0
        self.cost_r = 0.0

    # 基于用户的推荐
    # 根据对电影的评分计算用户之间的相似度
    def recommendByUser(self, userId):
        self.formatRate()
        # 推荐个数 等于 本身评分电影个数，用户计算准确率
        #此行代码有待考证
        #self.n = len(self.userDict[userId]) #推荐个数
        self.getNearestNeighbor(userId)
        self.getrecommandList(userId)
        self.getPrecision(userId)
        self.getRecall(userId)

    # 获取推荐列表
    def getrecommandList(self, userId):
        self.recommandList = []
        # 建立推荐字典
        recommandDict = {}
        for neighbor in self.neighbors:
            movies = self.userDict[neighbor[1]]
            for movie in movies:
                if (movie[0] in recommandDict):
                    recommandDict[movie[0]] += neighbor[0] #累加相似度
                else:
                    recommandDict[movie[0]] = neighbor[0]

        # 建立推荐列表
        for key in recommandDict:
            self.recommandList.append([recommandDict[key], key])
        self.recommandList.sort(reverse=True)
        self.recommandList = self.recommandList[:self.n] #截取十个
        print ('推荐列表长度:',len(self.recommandList))

    # 将ratings转换为userDict和ItemUser
    def formatRate(self):
        self.userDict = {}
        self.ItemUser = {}

        for i in self.ratings:
            # 评分最高为5 除以5 进行数据归一化
            temp = (i[1], float(i[2]) / 5)
            # 计算userDict {'1':[(1,5),(2,5)...],'2':[...]...}
            if (i[0] in self.userDict):
                self.userDict[i[0]].append(temp)
            else:
                self.userDict[i[0]] = [temp]
            # 计算ItemUser {'1',[1,2,3..],...}
            if (i[1] in self.ItemUser):
                self.ItemUser[i[1]].append(i[0])
            else:
                self.ItemUser[i[1]] = [i[0]]
        #print ('ratings格式化完成')

    # 找到某用户的相邻用户
    def getNearestNeighbor(self, userId):
        neighbors = []
        self.neighbors = []
        # 获取userId评分的电影都有那些用户也评过分
        for i in self.userDict[userId]:
            for j in self.ItemUser[i[0]]:
                if (j != userId and j not in neighbors):
                    neighbors.append(j)
        # 计算这些用户与userId的相似度并排序
        for i in neighbors:
            dist = self.getCost(userId, i)
            self.neighbors.append([dist, i])
        # 排序默认是升序，reverse=True表示降序
        self.neighbors.sort(reverse=True)
        self.neighbors = self.neighbors[:self.k]


    # 格式化userDict数据
    def formatuserDict(self, userId, l):
        user = {}
        for i in self.userDict[userId]:
            user[i[0]] = [i[1], 0]
        for j in self.userDict[l]:
            if (j[0] not in user):
                user[j[0]] = [0, j[1]]
            else:
                user[j[0]][1] = j[1]
        return user

    # 计算余弦距离
    def getCost(self, userId, l):
        # 获取用户userId和l评分电影的并集
        # {'电影ID'：[userId的评分，l的评分]} 没有评分为0
        embedding_temp = []
        for i in self.embeddings:
            if i[0] == userId:
                embedding_temp.append(i)
        for i in self.embeddings:
            if i[0] == l:
                embedding_temp.append(i)
        #print ('获取embedding完成',userId,l)
        #截取嵌入
        embedding_temp[0] = embedding_temp[0][1:]
        embedding_temp[1] = embedding_temp[1][1:]
        #转换类型
        embedding_temp[0]=self.vert_str2float(embedding_temp[0])
        embedding_temp[1] = self.vert_str2float(embedding_temp[1])
        #向量化
        vector_a = np.array(embedding_temp[0]).reshape(128,1)
        vector_b = np.array(embedding_temp[1]).reshape(128,1)

        num = vector_a.T.dot(vector_b)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        sim = float(sim)
        return sim

    #转换数据类型
    def vert_str2float(self,temp):
        temp1 = []
        for i in temp:
            temp1.append(float(i))
        return temp1

    # 推荐的准确率
    def getPrecision(self, userId):
        user_movie = [i[0] for i in self.userDict[userId]]
        recommand = [i[1] for i in self.recommandList]
        count = 0.0

        for i in recommand:
            if (i in user_movie):
                count += 1.0
        self.cost_p = count / len(recommand)
    # 推荐的召回率
    def getRecall(self, userId):
        user_movie = [i[0] for i in self.userDict[userId]]
        recommand = [i[1] for i in self.recommandList]
        count = 0.0

        for i in user_movie:
            if (i in recommand):
                count += 1.0
        self.cost_r = count / len(user_movie)


    # 显示推荐列表
    def showTable(self):
        neighbors_id = [i[1] for i in self.neighbors]
        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(["t", "t", "t", "t"])
        table.set_cols_align(["l", "l", "l", "l"])
        rows = []
        rows.append([u"movie ID", u"Name", u"release", u"from userID"])
        for item in self.recommandList:
            fromID = []
            for i in self.movies:
                if i[0] == item[1]:
                    movie = i
                    break
            for i in self.ItemUser[item[1]]:
                if i in neighbors_id:
                    fromID.append(i)
            movie.append(fromID)
            rows.append(movie)
        table.add_rows(rows)
        print(table.draw())


# 获取数据
def embedding_readFile(filename):
    files = open(filename, "r")
    #得到用户嵌入向量
    embeddings = []
    user_id = []
    for line in files.readlines():
        item = line.strip().split(' ')
        embeddings.append(item)
        user_id.append(int(float(item[0])))
    user_id.sort()
    return embeddings,user_id
def readFile_movie(filename):
    files = open(filename, "r")
    # 如果读取不成功试一下
    #files = open(filename, "r", encoding="utf-8")
    data = []
    for line in files.readlines():
        item = line.strip().split("::")
        data.append(item)
    return data
def readFile(filename):
    #files = open(filename, "r")
    # 如果读取不成功试一下
    files = open(filename, "r")
    data = []
    for line in files.readlines():
        item = line.strip().split("\t")
        data.append(item)
    return data

start = time.clock()
movies = readFile_movie('../CF/fake_data/movies.txt')
embeddings,user_id = embedding_readFile('../data/Yelp/embeddings/ubu_0.8.embedding')
ratings = readFile('../data/Yelp/ub_0.8.train')
demo = CF_embedding(movies, ratings, embeddings, k = 20,n = 10)
avg_p = 0.0
avg_r = 0.0
for i in user_id:
    demo.recommendByUser(str(i))
    print ('用户',str(i),'的相关信息')
#print("推荐列表为：")
#demo.showTable()
#print("处理的数据为%d条" % (len(demo.ratings)))
    print("准确率： %.2f %%" % (demo.cost_p*100))
    avg_p = avg_p + demo.cost_p
    print("召回率： %.2f %%" % (demo.cost_r*100))
    avg_r = avg_r + demo.cost_r
print ((avg_p/len(user_id))*100)
print ((avg_r/len(user_id))*100)
#end = time.clock()
#print("耗费时间： %f s" % (end - start))
