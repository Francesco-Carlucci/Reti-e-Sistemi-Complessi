import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import scipy.interpolate
import string
import multiprocessing
import random
import pickle

#@profile #used to speed up graph generation and neighbour polling, comment if running
def expPerformer(pin,pout,minority, nexp,L2,N):
    minwin=0
    for i in range(0, nexp):
        #random.shuffle(L2)  # shuffle intentions
        G = nx.generators.community.stochastic_block_model([minority,N-minority],[[pin,pout],[pout,pin]])

        vote = [0, 0]  # array of vote counter: 0 is minority, 1 is majority

        #A = nx.to_numpy_matrix(G) #too slow
        #majvet=A*(np.transpose(np.array([L2])))

        for i in range(0, N):
            neighb = G.neighbors(i)  # iterator on the neighbor
            vet = [L2[n] for n in neighb]
            maj = sum(vet)  # sum of 1 and 0 equals the number of majority voters in the neighborhood
            min = len(vet) - maj  # len(vet) is the number of neighbor

            if L2[i] == 0 and ((maj == min) or (maj == min + 1)):
                vote[0] += 1
            elif L2[i] == 1 and ((maj == min) or (maj + 1 == min)):
                vote[1] += 1
            # print("voti minoranza: ",vote[0],"voti maggioranza: ",vote[1])
        if vote[0] > vote[1]:
            minwin += 1    #minority win
    return minwin

def main():
    N = 100     #number of nodes
    step = 0.01
    nexp = 100
    matdim=int(1 / step)+1
    #prob = np.zeros((matdim,matdim))
    pvalues = np.arange(0.0, 1.0+step, step)
    #idin = 0

    log = open('probvaluesSliced.txt', 'wb')

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
    plt.rcParams.update({'mathtext.default': 'regular'})
    axs[1].set_xlabel('$p_{in}$')
    axs[0].set_ylabel('$p_{out}$')
    col = 0

    for minority in [20, 30, 40]:
        L2=np.ndarray.tolist(np.zeros((1,minority)))[0]
        L2=L2+np.ndarray.tolist(np.ones((1,N-minority)))[0] #intention of vote
        idin=0
        prob = np.zeros((matdim,matdim))
        for pin in pvalues:       #prob within communities, x axis
            idout=0
            for pout in pvalues[0:idin+10]:  #prob between communities, y axis
                minwin = 0
                numWorkers =multiprocessing.cpu_count()
                with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
                    futures = [executor.submit(expPerformer, pin,pout,minority, int(nexp / numWorkers),L2,N) for i in range(numWorkers)]
                    for future in concurrent.futures.as_completed(futures):
                        minwin += future.result()

                prob[idin,idout] = minwin / nexp
                #print("Probabilit?? di vittoria: ", prob[idin,idout], "con pin e pout grafo: ", pin,pout)
                idout +=1
            idin += 1
        title = '(' + string.ascii_letters[col] + ') N\u208B=' + str(minority) + '%'
        axs[col].title.set_text(title)
        points = np.transpose([np.tile(pvalues, matdim), np.repeat(pvalues, matdim)])
        #plt.scatter(points[:, 1], points[:, 0], c=prob)

        pickle.dump(prob,log)
        np.savetxt("probvaluesSliced.csv", prob, delimiter=",")

        sc=axs[col].scatter(points[:, 1], points[:, 0], c=prob.flatten()) #flatten needed to run on jupyter (different matplotlib version)
        col += 1
    plt.colorbar(sc)
    plt.savefig("StochasticBlock_toSlice.eps",format='eps')

    plt.show()
    log.close()
    """
    grid_x, grid_y = np.mgrid[0:1:(10/step)*1j,0:1:(10/step)*1j]
    probflat = np.reshape(prob, (matdim * matdim, 1))
    grid=scipy.interpolate.griddata(points, probflat, (grid_x,grid_y), method='linear')
    plt.imshow(grid, origin='lower', extent=(0,1,0,1))
    plt.show()
    """
if __name__=='__main__':
    main()
