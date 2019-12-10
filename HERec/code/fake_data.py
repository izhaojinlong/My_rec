import random
sample1 = ''.join(random.sample(['z','y','x','w','v','u','t','s','r','q','p',
                                 'o','n','m','l','k','j','i','h','g','f',
                                 'e','d','c','b','a'], 5))
print (sample1)

with open('movies.txt','w') as filewrite:
    for i in range(1,14285):
        sample1 = ''.join(random.sample(
            ['z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r', 'q', 'p',
             'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', 'f',
             'e', 'd', 'c', 'b', 'a'], 5))
        filewrite.writelines(str(i)+'::'+sample1+'::'+sample1+'A'+'\n')