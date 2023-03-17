import argparse
import json
import numpy as np

from optim.Grammar import Grammar
from optim.Lexicon import Lexicon
from optim.Phonology import SPE
from optim.Inventory import Inventory


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
    clxs, srs, ns = zip(*content)

    ## Generate the clxs by splitting by the delimiter ":"
    clxs = [tuple(clx.split(":")) for clx in clxs]

    ## Generate the lxs by getting all the unique lxs
    lxs = sorted(set(lx for clx in clxs for lx in clx))

    ## Convert the tuple of SRs to a list of SRs
    srs = list(srs)

    ## Convert the tuple of observations to a list of observations
    ns = [float(n) for n in ns]

    ## Create a dictionary to store all of the variables
    data = {"lxs": lxs, "clxs": clxs, "srs": srs, "ns": ns}

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
    rules = {"mnames": rnames, "mdefs": rdefs}

    return rules


def load_lexicon(filename):
    """Loads in lexicon file and returns the initial UR hypothesis for each lexeme
    in the space
    """

    ## Read in the lexicon
    prefix = "./parameters/lexicon/"
    content = load_content(f"{prefix}{filename}")

    ## Separate the list into the lexemes, lexical items, and underlying forms
    init_lxs, init_clxs, init_urs = zip(*content)

    ## Sort the list of lexemes
    init_lxs = sorted(init_lxs)

    ## Split the string underlying forms and contexts into separate lists
    init_clxs = [[tuple(x.split(":")) for x in cx.split(".")] for cx in init_clxs]
    init_urs = [[tuple(u.split(":")) for u in ur.split(".")] for ur in init_urs]

    ## Create a dictionary to store all of the variables to pass the model
    return init_lxs, init_clxs, init_urs


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                GRAMMAR INSTANTIATION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def process(data, inventory, lexicon, mappings):

    ## Load and initialize data parameters
    lm = data["lambda"]
    d = load_data(data["fn"])

    ## Load inventory
    i = load_inventory(inventory["fn"])

    ## Load and initialize mappings
    mp = {k: v for k, v in mappings.items() if k != "fn"}
    m = load_rules(mappings["fn"])

    ## Load the lexicon
    lp = {k: v for k, v in lexicon.items() if k != "fn"}
    l = load_lexicon(lexicon["fn"])

    ## Initialize the model
    ## G = Grammar()

    return G


def initialize():
    """Loads the parameters filename and populates a ``Grammar``
    with the provided parameters. Returns the ``Grammar`` object
    """

    ## Load the arguments from the command-line
    args = parse_args()

    ## Retrieve the parameters and process the variables
    params = load(args.fp)
    Grammar = process(**params)

    return G
