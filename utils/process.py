import numpy as np
import json
import argparse
from models.Grammar import Grammar


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                    ARGUMENT PARSER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        nargs="?",
        type=str,
        dest="fp",
        help="Filename containing the model parameters",
        required=True,
    )
    parser.add_argument(
        "-m",
        nargs="?",
        type=str,
        dest="fm",
        help="Filename containing the MCMC parameters",
        required=True,
    )
    args = parser.parse_args()
    return args


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                    GENERIC LOADER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def jload(filename: str) -> dict:
    with open(filename) as f:
        content = json.load(f)
    return content


def aload(filename: str) -> list:
    with open(filename) as f:
        content = [c.strip().split(",") for c in f.readlines()]
    return content


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                    PARAMETER LOADER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def process_obs(filename):
    """Loads in the surface data given a filename. Returns the lexemes, lexical
    items, and number of observations for each lexical item.

    (1) filename (str): name of the file to be read
    """

    ## Read in the data file
    prefix = "./parameters/data/"
    content = aload(f"{prefix}{filename}")

    ## Unzip the outputs and number of observations
    *out, nbs = zip(*content)

    ## Sort and split the outputs into lexical items and surface forms
    cxs, obs = zip(*sorted(zip(*out)))

    ## Generate the lexical items
    cxs = np.array([tuple(cx.split(":")) for cx in cxs], dtype=object)

    ## Generate all the lexemes
    lxs = np.array(sorted(set(lx for cx in cxs for lx in cx)))

    ## Convert the tuple of surface forms to a numpy array
    obs = np.array(obs)

    ## Convert the tuple of number of observations to a numpy array
    nbs = np.array([float(n) for n in nbs])

    ## Create a dictionary to store all of the variables
    data = (lxs, cxs, obs, nbs)

    return data


def process_inv(filename):
    """Loads in and reads an inventory file. Returns the tokens, features, and feature
    configurations for each token as numpy arrays.

    (1) filename (str): name of the file to be read
    """

    ## Read in the inventory file
    prefix = "./parameters/inventory/"
    content = aload(f"{prefix}{filename}")

    ## Convert to numpy array for easy indexing
    content = np.array(content)

    ## Separate the matrix into vectors / matrices for each of the variables
    tokens = content[:, 0][1:]
    feats = content[0, :][1:]
    configs = content[1:, 1:].astype(int)

    ## Create a dictionary to store all of the variables
    inventory = (tokens, feats, configs)

    return inventory


def process_rls(filename):
    """Loads in rules file. Returns the rule names and definitions in feature notation.

    (1) filename (str): name of the file to be read
    """

    ## Read in the inventory
    prefix = "./parameters/mappings/"
    content = aload(f"{prefix}{filename}")

    ## Convert to numpy array for easy indexing
    content = np.array(content)

    ## Check if the first entry is a number; if so, assume that there are 
    ## rule groupings
    grouped = content[0, 0].isnumeric()

    ## If it is grouped, separate the matrix into vectors / matrices 
    ## for each of the variables
    if grouped:
        groups = content[:, 0]
        rnames = content[:, 1]
        rdefs = content[:, 2:]
        ids = {}
        for gr, rn in zip(groups, rnames):
            ids.setdefault(gr, []).append((rn))

    ## Otherwise, assume each rule belongs in its own group
    else:
        rnames = content[:, 0]
        rdefs = content[:, 1:]
        ids = {gr: [rn] for gr, rn in enumerate(rnames)}

    ## Create a dictionary to store all of the variables
    rules = (ids, rnames, rdefs)

    return rules


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                PARAMETER INSTANTIATION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def process_parameters(obs, inv, map):

    ## Load the data parameters
    d = process_obs(obs)

    ## Load mappings
    m = process_rls(map)

    ## Load inventory
    i = process_inv(inv)

    return d, m, i


def load_parameters():
    """Loads the parameters filename and populates a ``Grammar`` with the provided
    parameters. Returns the ``Grammar`` object and MCMC parameters for the algorithm
    """

    ## Load the arguments from the command-line
    args = parse_args()

    ## Load the parameters from the file
    grP = jload(args.fp)
    mcP = jload(args.fm)

    return grP, mcP
