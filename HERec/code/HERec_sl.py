#!/usr/bin/python
# encoding=utf-8
import numpy as np
import time
import random
from math import sqrt, fabs, log
import sys


class HNERec:
    def __init__(self, unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile,
                 steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v):
        self.unum = unum
        self.inum = inum
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta #矩阵分解学习率
        self.beta_e = beta_e
        self.beta_h = beta_h
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u #预测评分中的参数alpha
        self.reg_v = reg_v #预测评分中的参数beta

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, unum)
        print('Load user embeddings finished.')

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, inum)
        print('Load user embeddings finished.')

        self.R, self.T, self.ba = self.load_rating(trainfile, testfile)
        print('Load rating finished.')
        print('train size : ', len(self.R))
        print('test size : ', len(self.T))

        self.initialize();
        self.recommend();

    def load_embedding(self, metapaths, num):
        X = {}
        for i in range(num):
            X[i] = {}
        metapathdims = []

        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embeddings/' + metapath
            # print sourcefile
            with open(sourcefile) as infile:
                '''k为user_metapath_dims'''
                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)
                for i in range(num):
                    X[i][ctn] = np.zeros(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
                print('metapath ', metapath, 'numbers ', n)
            ctn += 1
        return X, metapathdims

    '''设置训练矩阵和测试矩阵'''
    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []
        ba = 0.0
        n = 0
        user_test_dict = dict()
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_train.append([int(user) - 1, int(item) - 1, int(rating)])
                ba += int(rating)
                n += 1
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_test.append([int(user) - 1, int(item) - 1, int(rating)])

        return R_train, R_test, ba

    def initialize(self):
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1 #融合矩阵分解中的
        self.H = np.random.randn(self.inum, self.userdim) * 0.1 #融合矩阵分解中的
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1 #矩阵分解中的用户潜向量
        self.V = np.random.randn(self.unum, self.ratedim) * 0.1 #矩阵分解中的项目潜向量,与U维度相同

        '''用户跟项目的元路径集'''
        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum

        '''每个用户都会有一组参数,每个用户在不同元路径中的表示'''
        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1
        '''每个项目都会有一组参数,每个项目在不同元路径中的表示'''
        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            self.Wv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

    '''计算融合后的用户向量,单个用户'''
    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            ui += self.pu[i][k] * (self.Wu[k].dot(self.X[i][k]) + self.bu[k])
        return ui
    '''计算融合后的项目向量,单个项目'''
    def cal_v(self, j):
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * (self.Wv[k].dot(self.Y[j][k]) + self.bv[k])
        return vj
    '''预测评分'''
    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        #维度暂时没有搞明白
        print ('print shape')
        print (self.U[i, :].shape)
        print (self.V[i, :].shape)
        print (ui.shape)
        print (self.H[j, :].shape)
        print (self.E[i, :].shape)
        print (vj.shape)
        print ('end print')
        #输出看一下
        return self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj)

    def maermse(self):
        m = 0.0
        mae = 0.0
        rmse = 0.0
        n = 0
        for t in self.T:
            n += 1
            i = t[0]
            j = t[1]
            r = t[2]
            r_p = self.get_rating(i, j)

            if r_p > 5: r_p = 5
            if r_p < 1: r_p = 1
            m = fabs(r_p - r)
            mae += m
            rmse += m * m
        mae = mae * 1.0 / n
        rmse = sqrt(rmse * 1.0 / n)
        return mae, rmse

    def recommend(self):
        mae = []
        rmse = []
        starttime = time.clock()
        perror = 99999
        cerror = 9999
        n = len(self.R)
        '''更新训练集的'''
        for step in range(steps):
            total_error = 0.0
            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += eij * eij
                '''更新矩阵分解的参数'''
                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]
                '''更新单个用户潜向量和单个项目潜向量'''
                self.U[i, :] -= self.delta * U_g
                self.V[j, :] -= self.delta * V_g

                ui = self.cal_u(i) #融合后的单个用户向量
                '''reg_u:预测评分时用户的权重'''
                '''Wu*X+bu的维度为userdim'''
                for k in range(self.user_metapathnum):
                    pu_g = self.reg_u * -eij * self.H[j, :].dot(
                        self.Wu[k].dot(self.X[i][k]) + self.bu[k]) + self.beta_p * self.pu[i][k]
                    '''Wu_g的维度为user_dim*user_metapath_dims'''
                    '''X的维度(unum,user_metapath_num,user_metapath_dims)'''
                    Wu_g = self.reg_u * -eij * self.pu[i][k] * np.array([self.H[j, :]]).T.dot(
                        np.array([self.X[i][k]])) + self.beta_w * self.Wu[k]
                    '''bu_g的维度为userdim'''
                    bu_g = self.reg_u * -eij * self.pu[i][k] * self.H[j, :] + self.beta_b * self.bu[k]

                    '''更新用户的参数,分别在不同的元路径中'''
                    # self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Wu[k] -= 0.1 * self.delta * Wu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g
                '''H_g是一个值'''
                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g


                vj = self.cal_v(j)
                '''reg_v:预测评分时项目的权重'''
                for k in range(self.item_metapathnum):
                    pv_g = self.reg_v * -eij * self.E[i, :].dot(
                        self.Wv[k].dot(self.Y[j][k]) + self.bv[k]) + self.beta_p * self.pv[j][k]
                    Wv_g = self.reg_v * -eij * self.pv[j][k] * np.array([self.E[i, :]]).T.dot(
                        np.array([self.Y[j][k]])) + self.beta_w * self.Wv[k]
                    bv_g = self.reg_v * -eij * self.pv[j][k] * self.E[i, :] + self.beta_b * self.bv[k]

                    # self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Wv[k] -= 0.1 * self.delta * Wv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g

            perror = cerror
            '''n是训练样本的数量'''
            cerror = total_error / n

            self.delta = self.delta * 0.93
            if (abs(perror - cerror) < 0.0001):
                break
            # print 'step ', step, 'crror : ', sqrt(cerror)
            MAE, RMSE = self.maermse()
            mae.append(MAE)
            rmse.append(RMSE)
            # print 'MAE, RMSE ', MAE, RMSE
            endtime = time.clock()
            # print 'time: ', endtime - starttime
        print('MAE: ', min(mae), ' RMSE: ', min(rmse))


if __name__ == "__main__":
    unum = 16239
    inum = 14284
    ratedim = 10
    userdim = 30
    itemdim = 10
    train_rate = 0.8  # float(sys.argv[1])

    user_metapaths = ['ubu', 'ubcibu', 'ubcabu']
    item_metapaths = ['bub', 'bcib', 'bcab']

    for i in range(len(user_metapaths)):
        user_metapaths[i] += '_' + str(train_rate) + '.embedding'
    for i in range(len(item_metapaths)):
        item_metapaths[i] += '_' + str(train_rate) + '.embedding'
    # user_metapaths = ['ubu_' + str(train_rate) + '.embedding', 'ubcibu_'+str(train_rate)+'.embedding', 'ubcabu_'+str(train_rate)+'.embedding']

    # item_metapaths = ['bub_'+str(train_rate)+'.embedding', 'bcib.embedding', 'bcab.embedding']
    trainfile = '../data/ub_' + str(train_rate) + '.train'
    testfile = '../data/ub_' + str(train_rate) + '.test'
    steps = 100
    delta = 0.02
    beta_e = 0.1
    beta_h = 0.1
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.01
    reg_u = 1.0
    reg_v = 1.0
    print('train_rate: ', train_rate)
    print('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print('max_steps: ', steps)
    print('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b',
          beta_b, 'reg_u', reg_u, 'reg_v', reg_v)

    HNERec(unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta,
           beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v)
