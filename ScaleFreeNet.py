import math
import random
import string
import time

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import multiprocessing

def powerlawGen(exp,size):
    return np.random.power(exp+1,size)/exp

def powerlawGen2(exp,size):
    p=np.random.random(size)
    x=p**(-1./exp)
    return x
@profile
def ScaleFreeGen(N,exp,h,intentions):  #custom strogatz paper algorithm implemented
    degsOrig = powerlawGen(exp, N)**(-1)
    degs = np.round(degsOrig).astype(int)
    """#checking prob distribution
    plt.hist(degsOrig)
    plt.hist(degs)
    print("media inversi: ",np.mean(degsOrig))
    print("media interi: ",np.mean(degs))
    plt.show()
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    majstack = []
    minstack = []
    minlen=0
    for i in range(N):
        for j in range(degs[i]):
            if intentions[i] == 0:
                minstack.append(i)
                minlen+=1
            elif intentions[i] == 1:
                majstack.append(i)
    random.shuffle(minstack)
    random.shuffle(majstack)
    hConnProb=np.random.random(minlen) #tiri casuali, uno per ogni arco da un nodo minoritario, lo connette con
                          #un altro di minoranza con probabilità h, generati prima perché numpy ottimizzato su vettori
    id=0
    while len(minstack) > 0:
        node1 = minstack.pop()
        if hConnProb[id] <= h:
            if len(minstack) != 0:
                G.add_edge(node1, minstack.pop())
        else:
            if len(majstack) != 0:
                G.add_edge(node1, majstack.pop())
        id+=1

    while len(majstack) > 1:
        node1 = majstack.pop()
        node2 = majstack.pop()
        G.add_edge(node1, node2)
    """
    colorMap = []  # color graph nodes according to their vote intention and draw the graph
    for i in intentions:
        if i == 0:
            colorMap.append('yellow')
        else:
            colorMap.append('purple')
    nx.draw(G, with_labels=True, node_color=colorMap)
    plt.show()
    """
    return G

def expPerformer(N,exp,h,intentions,nexp):
    minwin=0
    for i in range(0, nexp):

        G = ScaleFreeGen(N,minority,exp,h,intentions)

        vote = [0, 0]  # array of vote counter: 0 is minority, 1 is majority

        for i in range(0, N):
            neighb = G.neighbors(i)  # iterator on the neighbor
            vet = [intentions[n] for n in neighb]  # extract neighbor intentions
            maj = sum(vet)  # sum of 1 and 0 equals the number of majority voters in the neighborhood
            min = len(vet) - maj  # len(vet) is the number of neighbor

            if intentions[i] == 0 and ((maj == min) or (maj == min + 1)):
                vote[0] += 1
            elif intentions[i] == 1 and ((maj == min) or (maj + 1 == min)):
                vote[1] += 1
        if vote[0] > vote[1]:
            minwin += 1    #minority win
    return minwin

def main():
    N=100
    exp=2.5

    step = 0.01
    nexp = 10000
    prob = np.zeros(int(1 / step)+1)  # minority victory probability, vector to be filled
    hvalues = np.arange(0.0, 1+step, step)


    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
    axs[1].set_xlabel("h")
    axs[0].set_ylabel("Probability of minority winning")
    col=0

    for minority in [20, 30, 40]:
        intentions = np.ndarray.tolist(np.zeros((1, minority)))[0] \
                     + np.ndarray.tolist(np.ones((1, N - minority)))[0]  # vote intentions
        id=0
        for h in hvalues:
            minwin = 0

            numWorkers = multiprocessing.cpu_count()
            with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
                futures = [executor.submit(expPerformer,N,exp,h,intentions, int(nexp / numWorkers)) for i in range(numWorkers)]
                for future in concurrent.futures.as_completed(futures):
                    minwin += future.result()

            prob[id] = minwin / nexp
            print("Probabilità di vittoria: ", prob[id], "con h grafo: ", h)
            id += 1
        title='('+string.ascii_letters[col]+') N\u208B='+str(minority)+'%'
        axs[col].set_ylim(0,1)
        axs[col].title.set_text(title)
        axs[col].plot(hvalues, prob)
        col+=1
    plt.show()





if __name__=="__main__":
    main()