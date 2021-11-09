import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import multiprocessing

#@profile
def expPerformer(pin,pout,minority, nexp,L2,N):
    minwin=0
    for i in range(0, nexp):
        #random.shuffle(L2)  # shuffle intentions
        G = nx.generators.community.stochastic_block_model([minority,N-minority],[[pin,pout],[pout,pin]])

        vote = [0, 0]  # array of vote counter: 0 is minority, 1 is majority

        for i in range(0, N):
            neighb = G.neighbors(i)  # iterator on the neighbor
            vet = [L2[n] for n in neighb]
            maj = sum(vet)  # sum of 1 and 0 equals the number of majority voters in the neighborhood
            min = len(vet) - maj  # len(vet) is the number of neighbor

            if L2[i] == 0 and ((maj == min) or (maj == min + 1)):
                vote[0] += 1
            elif L2[i] == 1 and ((maj == min) or (maj + 1 == min)):
                vote[1] += 1
        if vote[0] > vote[1]:
            minwin += 1    #minority win
    return minwin

def TheoricFunc(poutVet,min,maj):
    res=[]
    for pout in poutVet:
        res.append(1-(1-(math.comb(maj,min)*(pout**min)*((1-pout)**(maj-min))+math.comb(maj,min-1)*(pout**(min-1))*((1-pout)**(maj-min+1))))**min)
    return res

def main():
    N = 10     #number of nodes
    #alfa=2/10
    step = 0.01
    nexp = 10000
    matdim=int(1 / step)+1
    prob = np.empty(matdim)
    print(prob.shape)
    pvalues = np.arange(0.0, 1.0+step, step)
    pin=1
    id=0

    for minority in [2,3,4]:  #usare alfa per adattarsi a differenti N
        #minority=int((alfa/(1+alfa))*N)
        idout = 0
        L2 = np.ndarray.tolist(np.zeros((1, minority)))[0]
        L2 = L2 + np.ndarray.tolist(np.ones((1, N - minority)))[0]  # intentions of vote
        for pout in pvalues:  #prob between communities, y axis
            minwin = 0

            numWorkers =multiprocessing.cpu_count()
            with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
                futures = [executor.submit(expPerformer, pin,pout,minority, int(nexp / numWorkers),L2,N) for i in range(numWorkers)]
                for future in concurrent.futures.as_completed(futures):
                    minwin += future.result()

            prob[idout] = minwin / nexp
            print("Probabilità di vittoria: ", prob[idout], "con pin e pout grafo: ", pin,pout)
            idout +=1
        np.savetxt("probvalues_b.csv", prob, delimiter=",")
        colors=['blue','purple','orange']
        #if minority==2:
        plt.plot(pvalues, prob, linewidth=2,color=colors[id])
        plt.plot(pvalues, TheoricFunc(pvalues,minority,N-minority),'b--',linewidth=0.5)
        id+=1
        """
        elif minority==3:
            plt.plot(pvalues, prob, linewidth=2, color='purple')
            plt.plot(pvalues, TheoricFunc(pvalues, minority, N - minority), 'b--',linewidth=0.5)
        elif minority==4:
            plt.plot(pvalues, prob, linewidth=2, color='orange')
            plt.plot(pvalues, TheoricFunc(pvalues, minority, N - minority), 'b--',linewidth=0.5)
        """
    plt.rcParams.update({'mathtext.default': 'regular'}) #use latex like input in matplot labels
    plt.ylabel("Probability of minority winning")
    plt.xlabel('$p_{out}$')
    plt.show()

if __name__=='__main__':
    main()