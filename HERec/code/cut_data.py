#!/user/bin/python
# encoding=utf8
import random

train_rate = 0.8
'''
#Yelp Dataset
#ub.txt表示用户/项目/评分,该文件划分训练集和测试集
R = []
with open('../data/Yelp/ub.txt', 'r') as infile:
    for line in infile.readlines():
        user, item, rating = line.strip().split('\t')
        R.append([user, item, rating])
random.shuffle(R)
train_num = int(len(R) * train_rate)
print ('Yelp train_num',train_num)
#写入训练集和测试集
with open('../data/Yelp/ub_' + str(train_rate) + '.train', 'w') as trainfile,\
     open('../data/Yelp/ub_' + str(train_rate) + '.test', 'w') as testfile:
     for r in R[:train_num]:
         trainfile.write('\t'.join(r) + '\n')
     for r in R[train_num:]:
         testfile.write('\t'.join(r) + '\n')

'''
#Douban movie
R1 = []
with open('../data/Douban_Movie/user_movie_process.dat', 'r') as f:
    for line in f.readlines():
        user, item, rating = line.strip().split('\t')
        R1.append([user,item,rating])
random.shuffle(R1)
train_num1 = int(len(R1)*train_rate)
print ('Douban_Movie train_num',train_num1)
#写入训练集和测试集
with open('../data/Douban_Movie/um_' + str(train_rate) + '.train','w') as f1,\
     open('../data/Douban_Movie/um_' + str(train_rate) + '.test','w') as f2:
    for r in R1[:train_num1]:
        f1.write('\t'.join(r)+ '\n')
    for r in R1[train_num1:]:
        f2.write('\t'.join(r) + '\n')


#Movielens Dataset
'''
R2 = []
with open('../data/Movielens/user_movie.dat', 'r') as f:
    for line in f.readlines():
        user, item, rating, timestamp = line.strip().split('\t')
        R2.append([user,item,rating])
random.shuffle(R2)
train_num2 = int(len(R2)*train_rate)
print ('Douban_Movie train_num',train_num2)
#写入训练集和测试集
with open('../data/Movielens/um_' + str(train_rate) + '.train','w') as f1,\
     open('../data/Movielens/um_' + str(train_rate) + '.test','w') as f2:
    for r in R2[:train_num2]:
        f1.write('\t'.join(r)+ '\n')
    for r in R2[train_num2:]:
        f2.write('\t'.join(r) + '\n')
'''