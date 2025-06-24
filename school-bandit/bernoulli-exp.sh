#!/bin/bash

dir="example-bern"
for r in {1..10}
do
   python3 run_atlucb.py -s $r -t 1000 -m 2 -e "bernoulli{'n':50}" > $dir/atlucb-$r.csv
   python3 postprocess.py -m 2 -e "bernoulli{'n':50}" -c $dir/atlucb-$r.csv -s prop_of_success >  $dir/atlucb-$r.prop_of_success
   python3 run_uniform.py -s $r -t 2000 -m 2 -e "bernoulli{'n':50}" > $dir/uniform-$r.csv
   python3 postprocess.py -m 2 -e "bernoulli{'n':50}}" -c $dir/uniform-$r.csv -s prop_of_success > $dir/uniform-$r.prop_of_success
   python3 run_bfts.py -s $r -t 2000 -m 2 -e "bernoulli{'n':50}" -p "beta{'alpha':.5,'beta':0.5}" > $dir/bfts-$r.csv
   python3 postprocess.py -m 2 -e "bernoulli{'n':50}" -c $dir/bfts-$r.csv -s prop_of_success > $dir/bfts-$r.prop_of_success
done

python3 plot/merge_results.py -d $dir -a "uniform,atlucb,bfts" -r 10 -T 2000 -t prop_of_success 

python3 plot/plot.py -d $dir -a "uniform,atlucb,bfts" -s 2000 -t prop_of_success -o $dir/out.png 
