from run_atlucb import run_atlucb
from run_uniform import run_uniform
from run_bfts import run_bfts
from postprocess import postprocess
from plot.merge_results import merge
from plot.plot import plot

from environments.csv_dist import csv_dist_bandit, csv_dist_means
from algorithms.posteriors import TDistribution

dir_ = "./school-exp"

replicates = 100
m = 2
time = 1000
stat = "prop_of_success"

def create_t_dist():
   return TDistribution(.5)

def csv_fn(algo, seed):
   return dir_ + "/" + algo + "-" + str(seed) + ".csv"

def pp_fn(algo, seed):
   return dir_ + "/" + algo + "-" + str(seed) + "." + stat

csv_dist_fn = "./brute-force-reward.csv"
for seed in range(1,replicates+1):
   bandit = csv_dist_bandit(csv_dist_fn)
   real_means = csv_dist_means(csv_dist_fn)
   
   with open(csv_fn("atlucb", seed), "w") as f:
      run_atlucb(seed, bandit, m, time, f)
   with open(pp_fn("atlucb", seed), "w") as f:
      postprocess(real_means, m, stat, csv_fn("atlucb", seed), f)
   
   with open(csv_fn("uniform", seed), "w") as f:
      run_uniform(seed, bandit, m, 2*time, f)
   with open(pp_fn("uniform", seed), "w") as f:
      postprocess(real_means, m, stat, csv_fn("uniform", seed), f)

   with open(csv_fn("bfts", seed), "w") as f:
      run_bfts(seed, create_t_dist, bandit, m, 2*time, f)
   with open(pp_fn("bfts", seed), "w") as f:
      postprocess(real_means, m, stat, csv_fn("bfts", seed), f)

algos = ["uniform","atlucb","bfts"]
merge(algos, 2*time, replicates, dir_, stat)
plot(algos, 2*time, dir_, stat, None, None, dir_+"/out.png")
   
