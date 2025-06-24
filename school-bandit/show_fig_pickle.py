from argparse import ArgumentParser

import matplotlib.pyplot as plt

import pickle

if __name__ == "__main__":
    parser = ArgumentParser(description="show_fig_pickle")
    parser.add_argument("-f", "--fn", dest="fn", type=str, required=True)
    args = parser.parse_args()

with open(args.fn, 'rb') as f:
    fig = pickle.load(f)
    fig.show()
    plt.show()
