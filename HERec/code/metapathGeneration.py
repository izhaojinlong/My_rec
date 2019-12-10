#!/usr/bin/python
#coding=utf-8
import sys
import numpy as np
import random

class metapathGeneration:

    def __init__(self, unum, bnum, conum, canum, cinum):
        self.unum = unum + 1 #用户数量
        self.bnum = bnum + 1 #公司数量
        self.conum = conum + 1 #称赞种类
        self.canum = canum + 1 #城市数量
        self.cinum = cinum + 1 #种类数量
        ub = self.load_ub('../data/Yelp/ub_0.8.train')
        #根据数据集生成不同的元路径并存储到文件中
        self.get_UBU(ub, '../data/Yelp/metapath/ubu_0.8.txt')
        self.get_UBCaBU(ub, '../data/Yelp/bca.txt', '../data/Yelp/metapath/ubcabu_0.8.txt')
        self.get_UBCiBU(ub, '../data/Yelp/bci.txt', '../data/Yelp/metapath/ubcibu_0.8.txt')
        self.get_BUB(ub, '../data/Yelp/metapath/bub_0.8.txt')
        self.get_BCiB('../data/Yelp/bci.txt', '../data/Yelp/metapath/bcib_0.8.txt')
        self.get_BCaB('../data/Yelp/bca.txt', '../data/Yelp/metapath/bcab_0.8.txt')

    def load_ub(self, ubfile):
        #构建用户项目矩阵，有评分的设为1，没有则为0
        ub = np.zeros((self.unum, self.bnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                ub[int(user)][int(item)] = 1

        print(ub,'\n加载ub关系完成')

        return ub

    def get_UBU(self, ub, targetfile):
        print ('UMU...')

        uu = ub.dot(ub.T)
        print (uu.shape)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print ('UBU元路径完成')
    
    def get_BUB(self, ub, targetfile):
        print ('MUM...')
        mm = ub.T.dot(ub)
        print (mm.shape)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print('BUB元路径完成')
    
    def get_BCiB(self, bcifile, targetfile):
        print ('BCiB..')

        bci = np.zeros((self.bnum, self.cinum))
        with open(bcifile) as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                bci[int(m)][int(d)] = 1

        mm = bci.dot(bci.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print ('BCiB元路径完成')

    def get_BCaB(self, bcafile, targetfile):
        print ('BCaB..')

        bca = np.zeros((self.bnum, self.canum))
        with open(bcafile) as infile:
            for line in infile.readlines():
                m, a,__ = line.strip().split('\t')
                bca[int(m)][int(a)] = 1

        mm = bca.dot(bca.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print ('BCaB元路径完成')

    def get_UBCaBU(self, ub, bcafile, targetfile):
        print ('UBCaBU...')

        bca = np.zeros((self.bnum, self.canum))
        with open(bcafile, 'r') as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                bca[int(m)][int(d)] = 1

        uu = ub.dot(bca).dot(bca.T).dot(ub.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print ('UBCaBU元路径完成')
    
    def get_UBCiBU(self, ub, bcifile, targetfile):
        print ('UBCiBU...')

        bci = np.zeros((self.bnum, self.cinum))
        with open(bcifile, 'r') as infile:
            for line in infile.readlines():
                m, a, _ = line.strip().split('\t')
                bci[int(m)][int(a)] = 1

        uu = ub.dot(bci).dot(bci.T).dot(ub.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print('UBCiBU元路径完成')

if __name__ == '__main__':
    #see __init__() 
    metapathGeneration(unum=16239, bnum=14284, conum=11, canum=511, cinum=47)
