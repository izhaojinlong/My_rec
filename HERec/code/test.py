# coding: utf-8

def embedding_readFile(filename):
    files = open(filename, "r")
    # 如果读取不成功试一下
    #files = open(filename, "r", encoding="iso-8859-15")
    data = []
    for line in files.readlines():
        item = line.strip().split(" ")
        data.append(int(float(item[0])))
        #print (item[0])
    data.sort()
    for i in range(len(data)):
        if data[i]!=i:
            print (i+1,data[i])
        else:
            print ('正确')
    return data

def formatRate(ratings):
    userDict = {}
    ItemUser = {}

    for i in ratings:
        # 评分最高为5 除以5 进行数据归一化
        temp = (i[1], float(i[2]) / 5)
        # 计算userDict {'1':[(1,5),(2,5)...],'2':[...]...}
        if (i[0] in userDict):
            userDict[i[0]].append(temp)
        else:
            userDict[i[0]] = [temp]
        # 计算ItemUser {'1',[1,2,3..],...}
        if (i[1] in ItemUser):
            ItemUser[i[1]].append(i[0])
        else:
            ItemUser[i[1]] = [i[0]]
    return ItemUser,userDict


embeddings = embedding_readFile('../data/Yelp/embeddings/ubu_0.8.embedding')

print ('1')







