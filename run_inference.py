import numpy as np
import json
import argparse
from models.Inventory import Inventory
from models.Lexicon import Lexicon
from models.Phonology import SPE, OT
from models.Grammar import Grammar
from models.MCMC import MCMC


def parse_args():
    """Reads in and parses arguments from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameters",
        "-p",
        nargs="?",
        type=str,
        dest="parameters_fn",
        help="Name of the parameters file.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        nargs="?",
        type=str,
        dest="mode",
        choices=["OT", "SPE"],
        default="SPE",
        help="The phonological mapping to use [OT/SPE].",
    )
    args = parser.parse_args()
    return args


def load_parameters(parameters_fn):
    """Loads parameters from JSON file"""
    with open(parameters_fn) as f:
        parameters = json.load(f)
    return parameters


def load_grammar_class(inventory_fn, lexicon_fn, mappings_fn, data_fn, mode):
    """Loads and initializes a Grammar class"""
    segs, feats, configs = load_inventory(inventory_fn)
    lxs, clxs, urs = load_lexicon(lexicon_fn)
    L = Lexicon(lxs, clxs, urs, segs, feats, configs)
    if mode == OT:
        cnames, cdefs = load_constraints(mappings_fn)
        M = OT(cnames, cdefs, segs, feats, configs)
    else:
        rnames, rdefs = load_rules(mappings_fn)
        M = SPE(rnames, rdefs, segs, feats, configs)
    clxs, srs, nobs = load_data(data_fn)
    G = Grammar(clxs, srs, nobs, L, M)
    return G


def load_inventory(inventory_fn):
    """Loads in inventory file and returns an Inventory object"""
    prefix = "./parameters/inventory/"
    with open(f"{prefix}{inventory_fn}") as f:
        inventory = f.readlines()
        inventory = [i.strip().split(",") for i in inventory]
        inventory = np.array(inventory)
    segs = inventory[:, 0][1:]
    feats = inventory[0, :][1:]
    configs = inventory[1:, 1:].astype(int)
    return segs, feats, configs


def load_lexicon(lexicon_fn):
    """Loads in a lexicon file and returns formatted
    lexemes and underlying forms
    """
    prefix = "./parameters/lexicon/"
    with open(f"{prefix}{lexicon_fn}") as f:
        lexicon = f.readlines()
        lexicon = [l.strip().split(",") for l in lexicon]
    lxs, clxs, urs = zip(*lexicon)
    lxs = list(lxs)
    clxs = [[tuple(xs.split(":")) for xs in clx.split(".")] for clx in clxs]
    urs = [tuple(ur.split(".")) for ur in urs]
    return lxs, clxs, urs


def load_constraints(constraints_fn):
    """Loads in a constraints file and returns a formatted
    constraint names and definitions
    """
    prefix = "./parameters/mappings/"
    with open(f"{prefix}{constraints_fn}") as f:
        constraints = f.readlines()
    return None


def load_rules(rules_fn):
    """Loads in rules file and returns a tuple of rule names and definitions"""
    prefix = "./parameters/mappings/"
    with open(f"{prefix}{rules_fn}") as f:
        rules = f.readlines()
        rules = [r.strip().split(",") for r in rules]
        rules = np.array(rules)
    rnames = rules[:, 0]
    rdefs = rules[:, 1:]
    return rnames, rdefs


def load_data(data_fn):
    """Loads in a surface form file and returns formatted
    lexemes and suurface forms
    """
    prefix = "./data/"
    with open(f"{prefix}{data_fn}") as f:
        lexicon = f.readlines()
        lexicon = [l.strip().split(",") for l in lexicon]
    clxs, srs, nobs = zip(*lexicon)
    clxs = [tuple(clx.split(":")) for clx in clxs]
    srs = list(srs)
    nobs = [int(nob) for nob in nobs]
    return clxs, srs, nobs


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                MAIN FUNCTION CALL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():
    """PARSE ARGUMENTS"""
    args = parse_args()
    parameters = load_parameters(args.parameters_fn)
    mode = args.mode

    """LOAD GRAMMAR CLASS"""
    inventory_fn = parameters["inventory_fn"]
    lexicon_fn = parameters["lexicon_fn"]
    mappings_fn = parameters["mappings_fn"]
    data_fn = parameters["surface_forms_fn"]
    G = load_grammar_class(inventory_fn, lexicon_fn, mappings_fn, data_fn, mode)

    """RUN MCMC MODEL"""
    iterations = parameters["iterations"]
    sampled_grammars = MCMC.gibbs_sampler(G, iterations)

    """PREDICTIVE POSTERIOR"""
    burned_in_sampled_grammars = MCMC.burn_in(sampled_grammars)
    predictive = MCMC.posterior_predictive(burned_in_sampled_grammars)

if __name__ == "__main__":
    main()
