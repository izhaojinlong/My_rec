import numpy as np
def str2float(list):
    for i in range(len(list)):
        for j in range(len(list[i])):
            if j==0:
                list[i][j] = int(float(list[i][j]))
            else:
                list[i][j] = float(list[i][j])
    return list

def float2str(list):
    for i in range(len(list)):
        for j in range(len(list[i])):
            if j==0:
                list[i][j] = str(int(list[i][j]))
            else:
                list[i][j] = str((list[i][j]))
    return list




def avg_embedding(list1,list2,list3,list4):
    list1 = np.array(list1)
    list2 = np.array(list2)
    list3 = np.array(list3)
    list4 = np.array(list4)
    temp = np.sum([list1, list2, list3,list4], axis=0)
    temp = temp/4
    return temp.tolist()

def avg_embedding_item(list1,list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    temp = np.sum([list1, list2], axis=0)
    temp = temp/2
    return temp.tolist()


def embeddings_process():
    # user_embedings and sorted
    user_embeddings = []
    with open('../../data/Douban_Movie/embeddings_process/umtmu_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                user_embeddings.append(line)
    f.close()
    user_embeddings.sort(key=lambda x: int(float(x[0])))


    user_embeddings2 = []
    with open('../../data/Douban_Movie/embeddings_process/umamu_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                user_embeddings2.append(line)
    f.close()
    user_embeddings2.sort(key=lambda x: int(float(x[0])))

    user_embeddings3 = []
    with open('../../data/Douban_Movie/embeddings_process/umu_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                user_embeddings3.append(line)
    f.close()
    user_embeddings3.sort(key=lambda x: int(float(x[0])))

    user_embeddings4 = []
    with open('../../data/Douban_Movie/embeddings_process/umdmu_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                user_embeddings4.append(line)
    f.close()
    user_embeddings4.sort(key=lambda x: int(float(x[0])))


    user_embeddings = str2float(user_embeddings)
    user_embeddings2 = str2float(user_embeddings2)
    user_embeddings3 = str2float(user_embeddings3)
    user_embeddings4 = str2float(user_embeddings4)
    fusion_user_embeddings=avg_embedding(user_embeddings,user_embeddings2,
                                         user_embeddings3,user_embeddings4)
    float2str(fusion_user_embeddings)

    print ('----------------------------------')

    item_embeddings = []
    with open('../../data/Douban_Movie/embeddings_process/mtm_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                item_embeddings.append(line)
    f.close()
    item_embeddings.sort(key=lambda x: int(float(x[0])))

    item_embeddings2 = []
    with open('../../data/Douban_Movie/embeddings_process/mum_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                item_embeddings2.append(line)
    f.close()
    item_embeddings2.sort(key=lambda x: int(float(x[0])))

    item_embeddings3 = []
    with open('../../data/Douban_Movie/embeddings_process/mam_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                item_embeddings3.append(line)
    f.close()
    item_embeddings3.sort(key=lambda x: int(float(x[0])))

    item_embeddings4 = []
    with open('../../data/Douban_Movie/embeddings_process/mdm_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                item_embeddings4.append(line)
    f.close()
    item_embeddings4.sort(key=lambda x: int(float(x[0])))



    item_embeddings = str2float(item_embeddings)
    item_embeddings2 = str2float(item_embeddings2)
    item_embeddings3 = str2float(item_embeddings3)
    item_embeddings4 = str2float(item_embeddings4)

    fusion_item_embeddings = avg_embedding(item_embeddings, item_embeddings2,
                                           item_embeddings3,item_embeddings4)

    float2str(fusion_item_embeddings)

    print('----------------------------------')


    print('ok')
    return 0
def padding_embedding():
    item = []
    item_embeddings = []
    with open('../../data/Douban_Movie/embeddings_process/mtm_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                item_embeddings.append(line)
                item.append(line[0])
    f.close()
    item_embeddings.sort(key=lambda x: int(float(x[0])))
    item.sort(key=lambda x: int(float(x)))

    item_embeddings3 = []
    with open('../../data/Douban_Movie/embeddings_process/mam_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                item_embeddings3.append(line)
    f.close()
    item_embeddings3.sort(key=lambda x: int(float(x[0])))

    item_embeddings4 = []
    with open('../../data/Douban_Movie/embeddings_process/mdm_0.8.embedding') as f:
        for line in f.readlines():
            line = line.strip('\n').split()
            if len(line) == 2:
                pass
            else:
                item_embeddings4.append(line)
    f.close()
    print ('---------')
padding_embedding()
embeddings_process()