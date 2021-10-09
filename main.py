import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from multiprocessing import Process,Value
import concurrent.futures

minwin=0
#@profile
def expPerformer(p, nexp):
    #print("Avanti!")
    minwin=0
    for i in range(0, nexp):
        random.shuffle(L2)  # shuffle intentions
        #intentions = dict(zip(L1, L2))  # radomly assigning intentions to graph nodes
        G = nx.generators.erdos_renyi_graph(N, p)
        #nx.set_node_attributes(G, intentions, "intentions")
        """
        colorMap=[] #color graph nodes according to their vote intention
        for i in L2:
            if i==0:
                colorMap.append('yellow')
            else:
                colorMap.append('purple')
        """
        vote = [0, 0]  # array of vote counter: 0 is minority, 1 is majority

        for i in range(0, N):
            neighb = G.neighbors(i)  # iterator on the neighbor
            vet = [L2[n] for n in neighb]  # extract neighbor intentions
            maj = sum(vet)  # sum of 1 and 0 equals the number of majority voters in the neighborhood
            min = len(vet) - maj  # len(vet) is the number of neighbor

            if L2[i] == 0 and ((maj == min) or (maj == min + 1)):
                vote[0] += 1
            elif L2[i] == 1 and ((maj == min) or (maj + 1 == min)):
                vote[1] += 1
            # print("voti minoranza: ",vote[0],"voti maggioranza: ",vote[1])
        if vote[0] > vote[1]:
            minwin += 1    #minority win
    #print("Finito")
    return minwin
    # nx.draw(G, with_labels=True, node_color=colorMap) #draw the colored graph
    # plt.show()

if __name__ == '__main__':
    N=100
    minority=20
    L1=range(0,N)        #node numbers
    L2=np.ndarray.tolist(np.zeros((1,minority)))[0]
    L2=L2+np.ndarray.tolist(np.ones((1,N-minority)))[0] #intention of vote

    step=0.005
    nexp=4000
    prob=np.zeros(int(1/step))      #minority victory probability, vector to be filled
    pvalues=np.arange(0.0,1,step)
    id=0

    start = time.time()
    for p in pvalues:
        minwin=0

        numWorkers=multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
            futures = [executor.submit(expPerformer, p,int(nexp/numWorkers)) for i in range(numWorkers)]
            for future in concurrent.futures.as_completed(futures):
                minwin+=future.result()

        #expPerformer(p,nexp) #not parallelized version
        prob[id]=minwin/nexp
        print("Probabilità di vittoria: ",prob[id],"con p grafo: ",p)
        id+=1
    end = time.time()
    print("Duration: ", end - start)
    plt.plot(pvalues,prob)
    plt.show()
