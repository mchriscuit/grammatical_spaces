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
    )
    args = parser.parse_args()
    return args


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                    PARAMETER LOADER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def load(filename):
    with open(filename) as f:
        content = json.load(f)
    return content


def load_content(filename):
    with open(filename) as f:
        content = [c.strip().split(",") for c in f.readlines()]
    return content


def load_data(filename):
    """Loads in the surface data given a filename. Returns the lexemes, lexical
    items, and number of observations for each lexical item. Due to dimensionality
    issue, all variables are returned as lists.
    """

    ## Read in the data file
    prefix = "./parameters/data/"
    content = load_content(f"{prefix}{filename}")
    *out, nbs = zip(*content)

    ## Sort and split the outputs
    cxs, srs = zip(*sorted(zip(*out)))

    ## Generate the clxs by splitting by the delimiter ":"
    cxs = np.array([tuple(cx.split(":")) for cx in cxs], dtype=object)

    ## Generate the string lxs by getting all the unique lxs
    lxs = np.array(sorted(set(lx for cx in cxs for lx in cx)))

    ## Convert the tuple of SRs to a list of SRs
    srs = np.array(srs)

    ## Convert the tuple of observations to a list of observations
    nbs = np.array([float(n) for n in nbs])

    ## Create a dictionary to store all of the variables
    data = {"lxs": lxs, "cxs": cxs, "srs": srs, "nbs": nbs}

    return data


def load_inventory(filename):
    """Loads in and reads an inventory file. Returns the tokens, features, and feature
    configurations for each token as numpy arrays.
    """

    ## Read in the inventory
    prefix = "./parameters/inventory/"
    content = np.array(load_content(f"{prefix}{filename}"))

    ## Separate the matrix into vectors / matrices for each of the variables
    tokens = content[:, 0][1:]
    feats = content[0, :][1:]
    configs = content[1:, 1:].astype(int)

    ## Create a dictionary to store all of the variables
    inventory = {"tokens": tokens, "feats": feats, "configs": configs}

    return inventory


def load_rules(filename):
    """Loads in rules file and returns a tuple of rule names and definitions"""

    ## Read in the inventory
    prefix = "./parameters/mappings/"
    content = np.array(load_content(f"{prefix}{filename}"))

    ## Separate the matrix into vectors / matrices for each of the variables
    rnames = content[:, 0]
    rdefs = content[:, 1:]

    ## Create a dictionary to store all of the variables
    rules = {"mnms": rnames, "mdfs": rdefs}

    return rules


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                GRAMMAR INSTANTIATION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def process_grammar(data, inventory, lexicon, mappings):

    ## Load the data parameters
    d = load_data(data["fn"])

    ## Load inventory
    i = load_inventory(inventory["fn"])

    ## Load the lexicon
    l = {k: v for k, v in lexicon.items() if k != "fn"}

    ## Load and initialize mappings
    m = load_rules(mappings["fn"])

    ## Initialize the model
    G = Grammar(i, l, m, **d)

    return G


def initialize():
    """Loads the parameters filename and populates a ``Grammar``
    with the provided parameters. Returns the ``Grammar`` object
    """

    ## Load the arguments from the command-line
    args = parse_args()

    ## Retrieve the GRAMMAR parameters and process the variables
    gparams = load(args.fp)
    G = process_grammar(**gparams)

    ## Retrieve the MCMC parameters and process the variables
    C = load(args.fm)

    return G, C
