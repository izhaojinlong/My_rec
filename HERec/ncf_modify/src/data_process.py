import numpy as np
import random
class Process():
    def __init__(self):
        self.user_embeddings = []
        self.item_embeddings = []
        self.user_vetor = []
        self.item_vetor = []
        self.user_vetor_float = np.zeros([13367,8], dtype = float)
        self.item_vetor_float = np.zeros([12677,8], dtype = float)

    def index_save(data,type):
        # 0 user_id
        # 1 item_id
        if type == 0:
            with open('user_index','w') as f:
                for i in range(len(data)):
                    f.write(str(data.get_values()[i][0])+' '+str(data.get_values()[i][1]))
                    f.write('\n')
        if type == 1:
            with open('item_index','w') as f:
                for i in range(len(data)):
                    f.write(str(data.get_values()[i][0])+' '+str(data.get_values()[i][1]))
                    f.write('\n')
        return 0

    def embeddings_process(self):
        #user_embedings and sorted
        user_embeddings = []
        with open('../../data/Movielens/embeddings/umtmu_0.8.embedding') as f:
            for line in f.readlines():
                line  = line.strip('\n').split()
                if len(line)==2:
                    pass
                else:
                    user_embeddings.append(line)
        f.close()
        user_embeddings.sort(key=lambda x: int(float(x[0])))
        self.user_embeddings = user_embeddings

        # item_embedings and sorted
        item_embeddings = []
        with open('../../data/Movielens/embeddings/mtm_0.8.embedding') as f:
            for line in f.readlines():
                line  = line.strip('\n').split()
                if len(line)==2:
                    pass
                else:
                    item_embeddings.append(line)
        f.close()
        item_embeddings.sort(key=lambda x: int(float(x[0])))
        self.item_embeddings = item_embeddings

        self.make_lack_embedings()
        return 0

    def make_lack_embedings(self):
        lack_embeddings = ['711','1122','1130','1307','1325','1352','1453','1458',
                           '1482','1493','1546','1571','1579','1583','1599','1604',
                           '1618','1626','1632','1640','1649','1661','1662','1665']
        temp_embeddings = []
        for i in lack_embeddings:
            temp=[]
            temp.append(i)
            for j in range(0,8):
                random_a = round(random.uniform(-1,0.2),8)
                temp.append(str(random_a))
            temp_embeddings.append(temp)
        for i in temp_embeddings:
            self.item_embeddings.append(i)

        self.sava_embedings()
        return 0

    def sava_embedings(self):
        self.item_embeddings.sort(key=lambda x: int(float(x[0])))
        self.user_embeddings.sort(key=lambda x: int(float(x[0])))
        with open('../../data/Movielens/embeddings/user.embeddings','w') as f:
            for line in self.user_embeddings:
                f.write(str(line) + '\n')
        f.close()
        with open('../../data/Movielens/embeddings/item.embeddings','w') as f:
            for line in self.item_embeddings:
                f.write(str(line) + '\n')
        f.close()
        self.match_index_change_embeddings_to_vextor()
        return 0

    def match_index_change_embeddings_to_vextor(self):
        #load index
        user_index = []
        with open('user_index','r') as f:
            for line in f.readlines():
                line = line.strip('\n').split()
                user_index.append(line)
        item_index = []
        with open('item_index', 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split()
                item_index.append(line)
        # match index
        temp = []
        for index_line in user_index:
            for embedding_line in self.user_embeddings:
                if index_line[0]==embedding_line[0]:
                    temp.append(embedding_line[1:9])
        self.user_vetor = temp
        temp = []
        for index_line in item_index:
            for embedding_line in self.item_embeddings:
                if index_line[0]==embedding_line[0]:
                    temp.append(embedding_line[1:9])
        self.item_vetor = temp

        self.save_vector()
        return 0

    def save_vector(self):
        with open('../../data/Movielens/embeddings/user.vector','w') as f:
            for line in self.user_vetor:
                f.write(str(line)+'\n')
        f.close()
        with open('../../data/Movielens/embeddings/item.vector','w') as f:
            for line in self.item_vetor:
                f.write(str(line)+'\n')
        f.close()


    def load_vector(self):
        with open('../../data/Movielens/embeddings/user.vector','r') as f:
            i = 0
            for line in self.user_vetor:
                s = np.array(self.to_float(line)).reshape((1,8))
                self.user_vetor_float[i] = s
                i = i+1
        f.close()
        np.save('../../data/Movielens/embeddings/user.npy', self.user_vetor_float)

        with open('../../data/Movielens/embeddings/item.vector','r') as f:
            i = 0
            for line in self.user_vetor:
                s = np.array(self.to_float(line)).reshape((1, 8))
                self.item_vetor_float[i] = s
                i = i + 1
        f.close()
        np.save('../../data/Movielens/embeddings/item.npy', self.item_vetor_float)


    def to_float(self,list):
        for i in range(len(list)):
            list[i]=float(list[i])
        return list




