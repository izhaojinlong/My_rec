#!/usr/bin/python
#coding=utf-8
import sys
import numpy as np
import random

class metapathGeneration_douban_movie:

    def __init__(self, usernum, movienum, groupnum, directornum, actornum, typenum):
        self.usernum = usernum + 1 #用户数量
        self.movienum = movienum + 1 #电影数量
        self.groupnum = groupnum + 1 #小组种类
        self.directornum = directornum + 1 #导演数量
        self.actornum = actornum + 1 #演员数量
        self.typenum = typenum + 1  # 类型数量
        #um = self.load_um('../data/Douban_Movie/um_0.8.train')
        um = self.load_um('../data/Douban_Movie/user_movie_process.dat')
        #根据数据集生成不同的元路径并存储到文件中
        #self.get_UMU(um, '../data/Douban_Movie/metapath/umu_0.8.txt')
        #self.get_UMDMU(um, '../data/Douban_Movie/movie_director.dat', '../data/Douban_Movie/metapath/umdmu_0.8.txt')
        #self.get_UMTMU(um, '../data/Douban_Movie/movie_type.dat', '../data/Douban_Movie/metapath/umtmu_0.8.txt')
        #self.get_UMAMU(um, '../data/Douban_Movie/movie_actor.dat', '../data/Douban_Movie/metapath/umamu_0.8.txt')
        #self.get_MUM(um, '../data/Douban_Movie/metapath/mum_0.8.txt')
        #self.get_MAM('../data/Douban_Movie/movie_actor.dat', '../data/Douban_Movie/metapath/mam_0.8.txt')
        self.get_MDM('../data/Douban_Movie/movie_director.dat', '../data/Douban_Movie/metapath/mdm_0.8.txt')
        #self.get_MTM('../data/Douban_Movie/movie_type.dat', '../data/Douban_Movie/metapath/mtm_0.8.txt')

    def load_um(self, umfile):
        #构建用户项目矩阵，有评分的设为1，没有则为0
        um = np.zeros((self.usernum, self.movienum))
        with open(umfile, 'r') as infile:
            for line in infile.readlines():
                user, movie, rating = line.strip().split('\t')
                um[int(user)][int(movie)] = 1
        print('\n加载um关系完成')
        return um

    def get_UMU(self, um, targetfile):
        print ('UMU...')

        uu = um.dot(um.T)
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

    def get_UMDMU(self, um, mdfile, targetfile):
        print('UMDMU...')

        md = np.zeros((self.movienum, self.directornum))
        with open(mdfile, 'r') as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                md[int(m)][int(d)] = 1

        uu = um.dot(md).dot(md.T).dot(um.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)

        print('UMDMU元路径完成')

    def get_UMTMU(self, um, mtfile, targetfile):
        print('UMTMU...')

        mt = np.zeros((self.movienum, self.typenum))
        with open(mtfile, 'r') as infile:
            for line in infile.readlines():
                m, t, _ = line.strip().split('\t')
                mt[int(m)][int(t)] = 1

        uu = um.dot(mt).dot(mt.T).dot(um.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)

        print('UMTMU元路径完成')

    def get_UMAMU(self, um, mafile, targetfile):
        print('UMAMU...')

        ma = np.zeros((self.movienum, self.actornum))
        with open(mafile, 'r') as infile:
            for line in infile.readlines():
                m, a, _ = line.strip().split('\t')
                ma[int(m)][int(a)] = 1

        uu = um.dot(ma).dot(ma.T).dot(um.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)

        print('UMAMU元路径完成')
    
    def get_MUM(self, um, targetfile):
        print ('MUM...')
        mm = um.T.dot(um)
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
    
    def get_MAM(self, mafile, targetfile):
        print ('MAM..')

        ma = np.zeros((self.movienum, self.actornum))
        with open(mafile) as infile:
            for line in infile.readlines():
                m, a, _ = line.strip().split('\t')
                ma[int(m)][int(a)] = 1

        mm = ma.dot(ma.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print ('MAM元路径完成')

    def get_MDM(self, mdfile, targetfile):
        print ('MDM..')

        md = np.zeros((self.movienum, self.directornum))
        with open(mdfile) as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                md[int(m)][int(d)] = 1

        mm = md.dot(md.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print ('MDM元路径完成')

    def get_MTM(self, mtfile, targetfile):
        print ('MAM..')

        mt = np.zeros((self.movienum, self.typenum))
        with open(mtfile) as infile:
            for line in infile.readlines():
                m, a, _ = line.strip().split('\t')
                mt[int(m)][int(a)] = 1

        mm = mt.dot(mt.T)
        print ('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print ('total = ', total)

        print ('MTM元路径完成')





if __name__ == '__main__':
    #see __init__() 
    metapathGeneration_douban_movie(usernum=13367, movienum=12677, groupnum=2753,
                                    directornum=2449, actornum=6311, typenum=38)
