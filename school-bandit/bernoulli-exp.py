from run_atlucb import run_atlucb
from run_uniform import run_uniform
from run_bfts import run_bfts
from postprocess import postprocess
from plot.merge_results import merge
from plot.plot import plot

from environments.bernoulli import bernoulli_bandit, bernoulli_means
from algorithms.posteriors import Beta

dir_ = "./bernoulli-exp"

replicates = 10
n = 50
m = 2
time = 1000
stat = "prop_of_success"

def create_beta():
   #beta posterior, from a Jeffreys' prior
   return Beta(.5,.5)

def csv_fn(algo, seed):
   return dir_ + "/" + algo + "-" + str(seed) + ".csv"

def pp_fn(algo, seed):
   return dir_ + "/" + algo + "-" + str(seed) + "." + stat

for seed in range(1,replicates+1):
   bandit = bernoulli_bandit(n)
   real_means = bernoulli_means(n)
   
   with open(csv_fn("atlucb", seed), "w") as f:
      run_atlucb(seed, bandit, m, time, f)
   with open(pp_fn("atlucb", seed), "w") as f:
      postprocess(real_means, m, stat, csv_fn("atlucb", seed), f)
   
   with open(csv_fn("uniform", seed), "w") as f:
      run_uniform(seed, bandit, m, 2*time, f)
   with open(pp_fn("uniform", seed), "w") as f:
      postprocess(real_means, m, stat, csv_fn("uniform", seed), f)

   with open(csv_fn("bfts", seed), "w") as f:
      run_bfts(seed, create_beta, bandit, m, 2*time, f)
   with open(pp_fn("bfts", seed), "w") as f:
      postprocess(real_means, m, stat, csv_fn("bfts", seed), f)

algos = ["uniform","atlucb","bfts"]
merge(algos, 2*time, replicates, dir_, stat)
plot(algos, 2*time, dir_, stat, None, None, dir_+"/out.png")
   
