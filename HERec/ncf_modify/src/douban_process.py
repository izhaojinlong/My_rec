
def del_source_one():
    one = {}.fromkeys(range(1,13368))
    list = []
    list_source = []
    for i in one:
        one[i]=0
    with open('../../data/Douban_Movie/user_movie.dat','r') as f:
        for line in f.readlines():
            list_source.append(line)
            line = int(str(line.strip().split()[0]))
            one[line]=one[line]+1
    f.close()

    for i in one:
        if one[i] == 1:
            list.append(i)
    print (len(list))
    for i in list:
        one.pop(i)
    print(len(one))

    with open('../../data/Douban_Movie/user_movie_process.dat','w') as f_new:
        for i in list_source:
            temp = int(str(i.strip().split()[0]))
            if temp not in list:
                f_new.write(i)
    f_new.close()

def del_moive_type():
    one = {}.fromkeys(range(1,12678))
    for i in one:
        one[i]=0
    with open('../../data/Douban_Movie/embeddings/mtm_0.8.embedding','r') as f:
        for line in f.readlines():
            if len(line.strip('\n').split())==2:
                pass
            else:
                line = int(str(line.strip().split()[0]))
                one[line]=one[line]+1
    f.close()

    for i in one:
        if one[i] !=1:
            print(i)

def del_user_embedding():
    one = {}.fromkeys(range(1,12678))
    for i in one:
        one[i]=0
    with open('../../data/Douban_Movie/embeddings/mtm_0.8.embedding','r') as f:
        for line in f.readlines():
            if len(line.strip('\n').split())==2:
                pass
            else:
                line = int(str(line.strip().split()[0]))
                one[line]=one[line]+1
    f.close()

    for i in one:
        if one[i] !=1:
            print(i)
#del_source_one()
#del_moive_type()
del_user_embedding()
