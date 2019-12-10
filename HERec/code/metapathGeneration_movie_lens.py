#!/usr/bin/python
#coding=utf-8
import sys
import numpy as np
import random

class metapathGeneration:

    def __init__(self, unum, mnum, agenum, ocnum, genum):
        self.unum = unum + 1 #用户数量
        self.mnum = mnum + 1 #电影数量
        self.agenum = agenum + 1 #年龄的数量
        self.ocnum = ocnum + 1 #职业数量
        self.genum = genum + 1 #电影种类数量
        um = self.load_um('../data/Movielens/um_0.8.train')
        #根据数据集生成不同的元路径并存储到文件中

        self.get_UMU(um, '../data/Movielens/metapath/umu_0.8.txt')
        self.get_MUM(um, '../data/Movielens/metapath/mum_0.8.txt')
        self.get_UMGMU(um, '../data/Movielens/movie_genre.dat',
                       '../data/Movielens/metapath/umgmu_0.8.txt')

        self.get_UOU('../data/Movielens/user_occupation.dat',
                     '../data/Movielens/metapath/uou_0.8.txt')
        self.get_UAU('../data/Movielens/user_age.dat',
                     '../data/Movielens/metapath/uau_0.8.txt')

    def load_um(self, umfile):
        #构建用户项目矩阵，有评分的设为1，没有则为0
        um = np.zeros((self.unum, self.mnum))
        with open(umfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                um[int(user)][int(item)] = 1
        print(um,'\n加载um关系完成')
        return um

    def get_UMU(self, um, targetfile):
        print ('UMU...')
        uu = um.dot(um.T)
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
        print ('UMU元路径完成')
    
    def get_MUM(self, um, targetfile):
        print ('MUM...')
        mm = um.T.dot(um)
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
        print('MUM元路径完成')

    def get_UMGMU(self, um, mgfile, targetfile):
        print ('UMGMU...')
        mg = np.zeros((self.mnum, self.genum))
        with open(mgfile, 'r') as infile:
            for line in infile.readlines():
                m, d= line.strip().split('\t')
                mg[int(m)][int(d)] = 1

        uu = um.dot(mg).dot(mg.T).dot(um.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print ('total = ', total)
        print ('UMGMU元路径完成')

    def get_UOU(self, uofile, targetfile):
        print ('UOU..')
        uo = np.zeros((self.unum, self.ocnum))
        with open(uofile) as infile:
            for line in infile.readlines():
                m, d = line.strip().split('\t')
                uo[int(m)][int(d)] = 1
        mm = uo.dot(uo.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)
        print ('UOU元路径完成')

    def get_UAU(self, uafile, targetfile):
        print ('UAU..')
        uo = np.zeros((self.unum, self.agenum))
        with open(uafile) as infile:
            for line in infile.readlines():
                m, d = line.strip().split('\t')
                uo[int(m)][int(d)] = 1
        mm = uo.dot(uo.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)
        print ('UAU元路径完成')



if __name__ == '__main__':
    metapathGeneration(unum=943, mnum=1682, agenum=8, ocnum=21, genum=18)
