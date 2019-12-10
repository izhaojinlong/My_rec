import numpy as np

class Process():
    def __init__(self):
        self.user_embeddings = []
        self.item_embeddings = []
        self.user_vetor = []
        self.item_vetor = []
        self.user_vetor_float = np.zeros([11100,128], dtype = float)
        self.item_vetor_float = np.zeros([12677,128], dtype = float)

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
        with open('../../data/Douban_Movie/embeddings/umtmu_0.8.embedding') as f:
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
        with open('../../data/Douban_Movie/embeddings/mtm_0.8.embedding') as f:
            for line in f.readlines():
                line  = line.strip('\n').split()
                if len(line)==2:
                    pass
                else:
                    item_embeddings.append(line)
        f.close()
        item_embeddings.sort(key=lambda x: int(float(x[0])))
        self.item_embeddings = item_embeddings

        self.sava_embedings()
        return 0

    def embeddings_process_2(self):
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

        user_embeddings = self.str2float(user_embeddings)
        user_embeddings2 = self.str2float(user_embeddings2)
        user_embeddings3 = self.str2float(user_embeddings3)
        user_embeddings4 = self.str2float(user_embeddings4)
        fusion_user_embeddings = self.avg_embedding(user_embeddings, user_embeddings2,
                                               user_embeddings3, user_embeddings4)
        self.float2str(fusion_user_embeddings)

        print('----------------------------------')

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

        item_embeddings = self.str2float(item_embeddings)
        item_embeddings2 = self.str2float(item_embeddings2)

        fusion_item_embeddings = self.avg_embedding_item(item_embeddings, item_embeddings2)

        self.float2str(fusion_item_embeddings)

        print('----------------------------------')

        print('ok')
        return fusion_user_embeddings,fusion_item_embeddings


    def sava_embedings(self):
        #self.item_embeddings.sort(key=lambda x: int(float(x[0])))
        #self.user_embeddings.sort(key=lambda x: int(float(x[0])))

        self.user_embeddings=[]
        self.item_embeddings=[]
        self.user_embeddings,self.item_embeddings = self.embeddings_process_2()
        '''
        with open('../../data/Douban_Movie/embeddings/user.embeddings','w') as f:
            for line in self.user_embeddings:
                f.write(str(line) + '\n')
        f.close()
        with open('../../data/Douban_Movie/embeddings/item.embeddings','w') as f:
            for line in self.item_embeddings:
                f.write(str(line) + '\n')
        f.close()
        '''
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
                    temp.append(embedding_line[1:129])
        self.user_vetor = temp
        temp = []
        for index_line in item_index:
            for embedding_line in self.item_embeddings:
                if index_line[0]==embedding_line[0]:
                    temp.append(embedding_line[1:129])
        self.item_vetor = temp

        self.save_vector()
        return 0

    def save_vector(self):
        with open('../../data/Douban_Movie/embeddings/user.vector','w') as f:
            for line in self.user_vetor:
                f.write(str(line)+'\n')
        f.close()
        with open('../../data/Douban_Movie/embeddings/item.vector','w') as f:
            for line in self.item_vetor:
                f.write(str(line)+'\n')
        f.close()


    def load_vector(self):
        with open('../../data/Douban_Movie/embeddings/user.vector','r') as f:
            i = 0
            for line in self.user_vetor:
                s = np.array(self.to_float(line)).reshape((1,128))
                self.user_vetor_float[i] = s
                i = i+1
        f.close()
        np.save('../../data/Douban_Movie/embeddings/user.npy', self.user_vetor_float)

        with open('../../data/Douban_Movie/embeddings/item.vector','r') as f:
            i = 0
            for line in self.user_vetor:
                s = np.array(self.to_float(line)).reshape((1, 128))
                self.item_vetor_float[i] = s
                i = i + 1
        f.close()
        np.save('../../data/Douban_Movie/embeddings/item.npy', self.item_vetor_float)


    def to_float(self,list):
        for i in range(len(list)):
            list[i]=float(list[i])
        return list

    def str2float(self,list):
        for i in range(len(list)):
            for j in range(len(list[i])):
                if j == 0:
                    list[i][j] = int(float(list[i][j]))
                else:
                    list[i][j] = float(list[i][j])
        return list

    def float2str(self,list):
        for i in range(len(list)):
            for j in range(len(list[i])):
                if j == 0:
                    list[i][j] = str(int(list[i][j]))
                else:
                    list[i][j] = str((list[i][j]))
        return list

    def avg_embedding(self,list1, list2, list3, list4):
        list1 = np.array(list1)
        list2 = np.array(list2)
        list3 = np.array(list3)
        list4 = np.array(list4)
        #temp = np.sum([list1, list2, list3, list4], axis=0)
        #temp = temp / 4
        temp = list1*0.5+list2*0+list3*0.5+list4*0
        return temp.tolist()

    def avg_embedding_item(self,list1, list2):
        list1 = np.array(list1)
        list2 = np.array(list2)
        temp = np.sum([list1, list2], axis=0)
        temp = temp / 2
        return temp.tolist()



