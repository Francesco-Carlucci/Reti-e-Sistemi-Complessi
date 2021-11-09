import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import concurrent.futures
import multiprocessing
import random
import string
from scipy import spatial#for networkx random geometric graph generation, reduces complexity to O(n)
#instead of =(n^2) when it has to calculate each node neighbours at a distance<r, that have to be connected
#@profile
def GeomRandGen(N,h,r,intentions):
    pos=dict()
    for i in range(N):
        y = random.random()
        if random.random()<=h:         #node is in its preferred half
            if intentions[i]==0:       #right half is minority
                x=random.uniform(0.5,1)
                pos[i]=(x,y)
            elif intentions[i]==1:       #left half is majority
                x=random.uniform(0,0.5)
                pos[i]=(x,y)
        else:
            if intentions[i]==0:       #right half is minority
                x = random.uniform(0, 0.5)
                pos[i]=(x,y)
            elif intentions[i]==1:       #left half is majority
                x = random.uniform(0.5, 1)
                pos[i]=(x,y)
    kdtree = spatial.KDTree(list(pos.values()))
    pairs = kdtree.query_pairs(r)
    G=nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(pairs)
    #G=nx.random_geometric_graph(N, r,pos=pos)
    """
    colorMap = []  # color graph nodes according to their vote intention and draw the graph
    for i in intentions:
        if i == 0:
            colorMap.append('yellow')
        else:
            colorMap.append('purple')
    nx.draw(G, with_labels=True, node_color=colorMap,pos=pos)
    plt.show()        #! Only works if in main thread: call before pool
    """
    return G
#@profile
def expPerformer(N,h,r,intentions, nexp):
    minwin=0
    for i in range(nexp):
        G = GeomRandGen(N,h,r,intentions)

        vote = [0, 0]  # array of vote counter: 0 is minority, 1 is majority

        for i in range(0, N):
            neighb = G.neighbors(i)  # iterator on the neighbor
            vet = [intentions[n] for n in neighb]
            maj = sum(vet)  # sum of 1 and 0 equals the number of majority voters in the neighborhood
            min = len(vet) - maj  # len(vet) is the number of neighbor

            if intentions[i] == 0 and ((maj == min) or (maj == min + 1)):
                vote[0] += 1
            elif intentions[i] == 1 and ((maj == min) or (maj + 1 == min)):
                vote[1] += 1
            # print("voti minoranza: ",vote[0],"voti maggioranza: ",vote[1])
        if vote[0] > vote[1]:
            minwin += 1    #minority win
    return minwin

def main():
    N = 100     #number of nodes
    minority = 30   #N- in the paper
    intentions = np.ndarray.tolist(np.zeros((1, minority)))[0] \
                 + np.ndarray.tolist(np.ones((1, N - minority)))[0]  # vote intentions

    #GeomRandGen(N,1,0.8,intentions) #used only to draw the graph

    step = 0.01
    nexp = 10000
    print("numero esperimenti: ",nexp)
    matdim=int(1 / step)+1
    prob = np.empty(matdim)
    hvalues = np.arange(0.5, 1.0+step/2, step/2)
    print(matdim,hvalues.shape)

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
    axs[1].set_xlabel("h")
    axs[0].set_ylabel("Probability of minority winning")
    col = 0

    for r in [0.2, 0.5, 0.7]:  #values of h below 0.5 behave simmetrically 0:high homophily->0.5: no homophily
        id = 0
        for h in hvalues:
            minwin = 0

            numWorkers = multiprocessing.cpu_count()
            with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
                futures = [executor.submit(expPerformer, N, h, r, intentions, int(nexp / numWorkers)) for i
                           in range(numWorkers)]
                for future in concurrent.futures.as_completed(futures):
                    minwin += future.result()

            prob[id] = minwin / nexp
            print("Probabilità di vittoria: ", prob[id], "con h grafo: ", h)
            id += 1
        title = '(' + string.ascii_letters[col] + ') r=' + str(r)
        axs[col].set_ylim(0, 1)
        axs[col].title.set_text(title)
        axs[col].plot(hvalues, prob)
        col += 1
    plt.show()

if __name__=='__main__':
    main()
