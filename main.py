from multiprocessing import Pool
from Viral import main, ViralMarketing
import snap

def helper(n):
    if n <= 500:
        num_sims = 200
    else:
        num_sims = 10
    G = snap.GenRMat(n, 5*n, 0.55, 0.228, 0.212)
    viral = main(G, "RMAT", num_trials=num_sims)
    G = snap.LoadEdgeList(snap.PNGraph, "G_er_{}.txt".format(n))
    viral = main(G, "ER", num_trials=num_sims)
    G = snap.LoadEdgeList(snap.PNGraph, "G_pa_{}.txt".format(n))
    viral = main(G, "PA", num_trials=num_sims)

if __name__ == '__main__':
    pool = Pool(processes=4)
    pool.map(helper, [200, 500, 1000, 5000])
