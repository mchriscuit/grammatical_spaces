import numpy as np
import os
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


def load_grammar_class(inventory, lexicon, mappings, surface_forms, mode):
    """Loads and initializes a Grammar class"""

    ## Load in inventory and inventory object
    inventory_fn = inventory["fn"]
    tokens, feats, configs = load_inventory(inventory_fn)

    ## Load in mappings and mapping object
    mappings_fn = mappings["fn"]
    if mode == OT:
        cnames, cdefs = load_constraints(mappings_fn)
        M = OT(cnames, cdefs, tokens, feats, configs)
    else:
        rnames, rdefs = load_rules(mappings_fn)
        M = SPE(rnames, rdefs, tokens, feats, configs)

    ## Load in data and lexicon
    surface_forms_fn = surface_forms["fn"]
    ur_prior = lexicon["ur_prior"]
    ur_proposal = lexicon["ur_proposal"]
    lxs, xclxs, clxs, srs, nobs = load_surface_forms(surface_forms_fn)
    L = Lexicon(lxs, xclxs, ur_prior, ur_proposal, tokens, feats, configs)

    ## Initialize UR hypothesis
    lexicon_fn = lexicon["fn"]
    if lexicon_fn:
        L.initialize_urs(load_lexicon(lexicon_fn))
    else:
        L.initialize_urs()

    ## Load and return grammar object
    G = Grammar(clxs, srs, nobs, L, M)
    return G


def load_inventory(inventory_fn):
    """Loads in inventory file and returns an Inventory object"""
    prefix = "./parameters/inventory/"
    with open(f"{prefix}{inventory_fn}") as f:
        inventory = f.readlines()
        inventory = [i.strip().split(",") for i in inventory]
        inventory = np.array(inventory)
    tokens = inventory[:, 0][1:]
    feats = inventory[0, :][1:]
    configs = inventory[1:, 1:].astype(int)
    return tokens, feats, configs


def load_lexicon(lexicon_fn):
    """Loads in lexicon file and returns the initial UR hypothesis
    for each lexeme in the space"""
    prefix = "./parameters/lexicon/"
    with open(f"{prefix}{lexicon_fn}") as f:
        lexicon = f.readlines()
        lexicon = [s.strip().split(",") for s in lexicon]
    init_lxs, init_clxs, init_urs, init_alns, lbs = zip(*lexicon)
    init_lxs = list(init_lxs)
    init_clxs = [[tuple(x.split(":")) for x in cx.split(".")] for cx in init_clxs]
    init_urs = [ur.split(".") for ur in init_urs]
    init_alns = [[np.array(list(op)) for op in aln.split(".")] for aln in init_alns]
    lbs = [int(lb) for lb in lbs]
    return init_lxs, init_clxs, init_urs, init_alns, lbs


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


def load_surface_forms(surface_forms_fn):
    """Loads in a surface form file and returns formatted
    lexemes and surface forms
    """
    prefix = "./data/"
    with open(f"{prefix}{surface_forms_fn}") as f:
        surface_forms = f.readlines()
        surface_forms = [s.strip().split(",") for s in surface_forms]
    clxs, srs, nobs = zip(*surface_forms)
    clxs = sorted(tuple(clx.split(":")) for clx in clxs)
    lxs = sorted(set(lx for clx in clxs for lx in clx))
    xclxs = [[clx for clx in clxs if lx in clx] for lx in lxs]
    xclxs = [[tuple(["PROTO"])] + cxs for cxs in xclxs]
    srs = list(srs)
    nobs = [int(nob) for nob in nobs]
    return lxs, xclxs, clxs, srs, nobs


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                MAIN FUNCTION CALL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():
    """PARSE ARGUMENTS"""
    args = parse_args()
    parameters = load_parameters(args.parameters_fn)
    mode = args.mode

    """LOAD GRAMMAR CLASS"""
    inventory = parameters["inventory"]
    lexicon = parameters["lexicon"]
    mappings = parameters["mappings"]
    surface_forms = parameters["surface_forms"]
    G = load_grammar_class(inventory, lexicon, mappings, surface_forms, mode)

    """RUN MCMC MODEL"""
    gs_iterations = parameters["gs_iterations"]
    mh_iterations = parameters["mh_iterations"]
    sampled_grammars, lx_acceptances = MCMC.gibbs_sampler(G, gs_iterations, mh_iterations)

    """PREDICTIVE POSTERIOR"""
    burned_in = MCMC.burn_in(sampled_grammars)
    post, pred = MCMC.posterior_predictive(burned_in)

    """SAVE TO FILE"""
    ## Generate new directory
    output_path = f"./output/gs{gs_iterations}-mh{mh_iterations}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## Write to file
    acceptances_fn = f"{output_path}/acceptances.csv"
    with open(acceptances_fn, "w") as f:
        f.write("lx,gs,mh,ur_old,post_old,ur_new,post_new,accepted\n")
        for lx, acceptances in lx_acceptances.items():
            for gsi, acceptance in enumerate(acceptances):
                for mhi, (ur_old, post_old, ur_new, post_new, accepted) in enumerate(acceptance):
                    f.write(f"{lx},{gsi},{mhi},{'-'.join(ur_old)},{post_old},{'-'.join(ur_new)},{post_new},{accepted}\n")

    posterior_fn = f"{output_path}/posterior.csv"
    with open(posterior_fn, "w") as f:
        f.write("clxs,mname,urs,pred_srs,obs_srs,prob\n")
        for grammar, prob in post.items():
            clxs, mnames, urs, pred_srs, obs_srs = grammar
            f.write(f"{clxs},{mnames},{urs},{pred_srs},{obs_srs},{prob}\n")

    predictive_fn = f"{output_path}/predictive.csv"
    with open(predictive_fn, "w") as f:
        f.write("clx,pred_sr,prob\n")
        for clx, pred_srs in pred.items():
            for pred_sr, prob in pred_srs.items():
                f.write(f"{clx},{pred_sr},{prob}\n")


if __name__ == "__main__":
    main()
