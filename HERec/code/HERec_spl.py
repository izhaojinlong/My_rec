#!/usr/bin/python
#encoding=utf-8
import numpy as np
import time
import random
from math import sqrt,fabs,log,exp
import sys

class HNERec:
    def __init__(self, unum, inum, ratedim, userdim, itemdim, user_metapaths,item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v):
        self.unum = unum
        self.inum = inum
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta #矩阵分解更新率
        self.beta_e = beta_e #矩阵分解更新的学习率
        self.beta_h = beta_h #矩阵分解更新的学习率
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u #预测评分中的参数
        self.reg_v = reg_v #预测评分中的参数

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        '''
        X的第一个维度是用户的数量,user_num,表示不同的用户
        第二个维度是不同用户元路径的不同维度及其表示
        第三个维度是1
        '''
        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, unum)
        print ('Load user embeddings finished.')
        '''
        Y与X基本相同,只是user改为item
        '''
        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, inum)
        print ('Load user embeddings finished.')
        '''得到训练矩阵,测试矩阵,ba未使用'''
        self.R, self.T, self.ba = self.load_rating(trainfile, testfile)
        print ('Load rating finished.')
        print ('train size : ', len(self.R))
        print ('test size : ', len(self.T))

        self.initialize();
        self.recommend();

    def load_embedding(self, metapaths, num):
    # X 代表用户嵌入
        X = {}
        for i in range(num):
            X[i] = {}
        metapathdims = []
    
        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/Yelp/embeddings/' + metapath
            #print sourcefile
            with open(sourcefile) as infile:
                '''得到不同用户元路径维度'''
                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)
                for i in range(num):
                    X[i][ctn] = np.zeros(k)

                n = 0
                '''得到不同用户元路径维度下的向量表示'''
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
                print ('metapath ', metapath, 'numbers ', n)
            ctn += 1
        return X, metapathdims

    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []
        ba = 0.0
        n = 0
        user_test_dict = dict()
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_train.append([int(user)-1, int(item)-1, int(rating)])
                ba += int(rating)
                n += 1
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_test.append([int(user)-1, int(item)-1, int(rating)])

        return R_train, R_test, ba

    def initialize(self):
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1 #融合矩阵分解中的项目向量
        self.H = np.random.randn(self.inum, self.userdim) * 0.1 #融合矩阵分解中的用户向量
        self.U = np.random.randn(self.unum, self.ratedim) * 0.1 #传统矩阵分解中用户的潜向量
        self.V = np.random.randn(self.inum, self.ratedim) * 0.1 #传统矩阵分解中项目的潜向量
        '''用户跟项目的路径集'''
        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum

        '''每个用户都会有一组参数,每个用户在不同元路径中的表示'''
        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            self.Wv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))
    '''融合后单个用户的向量'''
    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        '''计算用户所有元路径下的值'''
        for k in range(self.user_metapathnum):
            '''某个用户在某条元路径下的值'''
            ui += self.pu[i][k] * self.sigmod((self.Wu[k].dot(self.X[i][k]) + self.bu[k]))
        return self.sigmod(ui)
    '''融合后单个项目的向量,与用户类似'''
    def cal_v(self, j):
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * self.sigmod((self.Wv[k].dot(self.Y[j][k]) + self.bv[k]))
        return self.sigmod(vj)
    '''返回用户对项目的评分'''
    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
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

        for step in range(steps):
            total_error = 0.0
            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += eij * eij
                '''矩阵分解更新公式'''
                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]
                '''矩阵分解更新用户向量和项目向量'''
                self.U[i, :] -= self.delta * U_g
                self.V[j, :] -= self.delta * V_g

                '''与用户相关的更新'''
                '''Zs'''
                ui = self.cal_u(i)
                for k in range(self.user_metapathnum):
                    '''Zf'''
                    x_t = self.sigmod(self.Wu[k].dot(self.X[i][k]) + self.bu[k])
                    '''(14)第三个公式'''
                    pu_g = self.reg_u * -eij * (ui * (1-ui) * self.H[j, :]).dot(x_t) + self.beta_p * self.pu[i][k]
                    '''(14)第一个公式'''
                    Wu_g = self.reg_u * -eij * self.pu[i][k] * np.array([ui * (1-ui) * x_t * (1-x_t) * self.H[j, :]]).T.dot\
                        (np.array([self.X[i][k]])) + self.beta_w * self.Wu[k]
                    '''(14)第二个公式'''
                    bu_g = self.reg_u * -eij * ui * (1-ui) * self.pu[i][k] * self.H[j, :] * x_t * (1-x_t) + self.beta_b * self.bu[k]
                    #print pu_g
                    '''(10)'''
                    self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Wu[k] -= 0.1 * self.delta * Wu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g
                '''(11)'''
                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                '''与项目相关的更新'''
                vj = self.cal_v(j)
                for k in range(self.item_metapathnum):
                    y_t = self.sigmod(self.Wv[k].dot(self.Y[j][k]) + self.bv[k])
                    pv_g = self.reg_v * -eij * (vj * (1-vj) * self.E[i, :]).dot(y_t) + self.beta_p * self.pv[j][k]
                    Wv_g = self.reg_v * -eij  * self.pv[j][k] * np.array([vj * (1-vj) * y_t * (1 - y_t) * self.E[i, :]]).T.dot(np.array([self.Y[j][k]])) + self.beta_w * self.Wv[k]
                    bv_g = self.reg_v * -eij * vj * (1-vj) * self.pv[j][k] * self.E[i, :] * y_t * (1 - y_t) + self.beta_b * self.bv[k]

                    self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Wv[k] -= 0.1 * self.delta * Wv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.1 * self.E[i, :]

                self.E[i, :] -= self.delta * E_g

            perror = cerror
            cerror = total_error / n
            
            self.delta = 0.93 * self.delta

            if(abs(perror - cerror) < 0.0001):
                break
            #print 'step ', step, 'crror : ', sqrt(cerror)
            MAE, RMSE = self.maermse()
            mae.append(MAE)
            rmse.append(RMSE)
            #print 'MAE, RMSE ', MAE, RMSE
            endtime = time.clock()
            #print 'time: ', endtime - starttime
        print ('MAE: ', min(mae), ' RMSE: ', min(rmse))

if __name__ == "__main__":
    unum = 16239
    inum = 14284
    ratedim = 10
    userdim = 30
    itemdim = 10
    train_rate = 0.8#sys.argv[1]

    user_metapaths = ['ubu', 'ubcibu', 'ubcabu']
    item_metapaths = ['bub', 'bcib', 'bcab']

    for i in range(len(user_metapaths)):
        user_metapaths[i] += '_' + str(train_rate) + '.embedding'
    for i in range(len(item_metapaths)):
        item_metapaths[i] += '_' + str(train_rate) + '.embedding'

    #user_metapaths = ['ubu_' + str(train_rate) + '.embedding', 'ubcibu_'+str(train_rate)+'.embedding', 'ubcabu_'+str(train_rate)+'.embedding'] 
    
    #item_metapaths = ['bub_'+str(train_rate)+'.embedding', 'bcib.embedding', 'bcab.embedding']
    trainfile = '../data/Yelp/ub_'+str(train_rate)+'.train'
    testfile = '../data/Yelp/ub_'+str(train_rate)+'.test'
    steps = 100
    delta = 0.02
    beta_e = 0.1
    beta_h = 0.1
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.1
    reg_u = 1.0
    reg_v = 1.0
    print ('train_rate: ', train_rate)
    print ('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print ('max_steps: ', steps)
    print ('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b', beta_b, 'reg_u', reg_u, 'reg_v', reg_v)

    HNERec(unum, inum, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v)
