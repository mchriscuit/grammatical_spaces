import numpy as np
import re
import os
import json
import argparse
from optim.Inventory import Inventory
from optim.Lexicon import Lexicon
from optim.Phonology import SPE, OT
from optim.Grammar import Grammar
from optim.MCMC import MCMC


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                ARGUMENT PASSER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

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

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                PARAMETER LOADER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

def load_parameters(parameters_fn):
    """Loads parameters from JSON file"""
    with open(parameters_fn) as f:
        parameters = json.load(f)
    return parameters


def load_inventory(inventory_fn):
    """Loads in and reads an inventory file"""
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
    init_lxs, init_clxs, init_urs = zip(*lexicon)
    init_lxs = list(init_lxs)
    init_clxs = [[tuple(x.split(":")) for x in cx.split(".")] for cx in init_clxs]
    init_urs = [[tuple(u.split(":")) for u in ur.split(".")] for ur in init_urs]
    return init_lxs, init_clxs, init_urs


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

    ## Generate the clxs by splitting by the delimiter ":"
    clxs = [tuple(clx.split(":")) for clx in clxs]

    ## Generate the lxs by getting all the unique lxs in the space of clxs
    lxs = sorted(set(lx for clx in clxs for lx in clx))

    ## Generate the list of clxs containing each lx
    lx_clxs = [[clx for clx in clxs if lx in clx] for lx in lxs]
    lx_clxs = [[tuple(["PROTO"])] + cxs for cxs in lx_clxs]

    ## Convert the tuple of SRs to a list of SRs
    srs = list(srs)

    ## Convers the tuple of observations to a list of observations
    nobs = [int(nob) for nob in nobs]

    return lxs, lx_clxs, clxs, srs, nobs


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                GRAMMAR INSTANTIATION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def load_grammar_class(inventory, lexicon, mappings, surface_forms, mode):
    """Loads and initializes a Grammar class"""

    ## Load in inventory information
    inventory_fn = inventory["fn"]
    tokens, feats, configs = load_inventory(inventory_fn)

    ## Load in mappings information
    mappings_fn = mappings["fn"]

    ## If we are using an OT-based theory, retrieve the constraints
    ## and instantiate an OT object
    if mode == OT:
        cnames, cdefs = load_constraints(mappings_fn)
        M = OT(cnames, cdefs, tokens, feats, configs)

    ## Otherwise, we are using a rule-based theory. Retrieve the rules
    ## and instantiate an SPE object
    else:
        rnames, rdefs = load_rules(mappings_fn)
        M = SPE(rnames, rdefs, tokens, feats, configs)

    ## Load in surface forms
    surface_forms_fn = surface_forms["fn"]

    ## Generate the lexical information
    lxs, lx_clxs, clxs, srs, nobs = load_surface_forms(surface_forms_fn)
    lx_params = lexicon["params"]

    ## Instantiate a Lexicon object
    L = Lexicon(lxs, lx_clxs, lx_params, tokens, feats, configs)

    ## Initialize UR hypothesis
    lexicon_fn = lexicon["fn"]
    L.initialize_urs(load_lexicon(lexicon_fn))

    ## Instantiate a Grammar object
    lm = surface_forms["lambda"]
    G = Grammar(clxs, srs, nobs, lm, L, M)

    return G


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                MAIN FUNCTION CALL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():
    """PARSE ARGUMENTS"""
    args = parse_args()
    params = load_parameters(args.parameters_fn)
    mode = args.mode

    """LOAD GRAMMAR CLASS"""
    inventory = params["inventory"]
    lexicon = params["lexicon"]
    mappings = params["mappings"]
    surface_forms = params["surface_forms"]
    G = load_grammar_class(inventory, lexicon, mappings, surface_forms, mode)

    """ ==================== RUN MODEL ===================================="""
    gs_iterations = params["MCMC"]["gs_iterations"]
    mh_iterations = params["MCMC"]["mh_iterations"]
    anneal = params["MCMC"]["anneal"]
    sg, lx_as = MCMC.gibbs_sampler(G, anneal, gs_iterations, mh_iterations)

    """ ==================== BURN IN ======================================"""
    bi = MCMC.burn_in(sg)
    post, pred = MCMC.posterior_predictive(bi)

    """ ==================== SAVE TO FILE ================================="""
    ## Retrieve hyperparameters for printing
    lm = surface_forms["lambda"]
    psi = lexicon["params"]["psi"]
    phi = lexicon["params"]["phi"]
    alpha = lexicon["params"]["alpha"]

    ## Generate new directory
    output_path = f"./output/"
    output_path += f"{re.sub('.csv', '', surface_forms['fn'])}"
    output_path += f"-gs{gs_iterations}-mh{mh_iterations}"
    output_path += f"-lamda{lm}-psi{psi}-phi{phi}-alpha{alpha}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## Write to file
    acceptances_fn = f"{output_path}/acceptances.csv"
    with open(acceptances_fn, "w") as f:
        f.write("gs,")
        f.write("lx,")
        f.write("mhyp,")
        f.write("pro_ur,")
        f.write("cxt_ur\n")
        for lx, g in lx_as.items():
            for gsi, (m, u) in enumerate(g):
                f.write(f"{gsi},")
                f.write(f"{lx},")
                f.write(f"{'-'.join(m)},")
                f.write(f"{'.'.join(['-'.join(p) for p in u.keys()])},")
                f.write(f"{'.'.join(u.values())}\n")

    posterior_fn = f"{output_path}/posterior.csv"
    with open(posterior_fn, "w") as f:
        f.write("clxs,")
        f.write("mhyp,")
        f.write("urs,")
        f.write("pred_srs,")
        f.write("obs_srs,")
        f.write("prob\n")
        for g, p in sorted(post.items(), key=lambda kv: kv[1], reverse=True):
            c, m, u, prd, obs = g
            f.write(f"{c},")
            f.write(f"{m},")
            f.write(f"{u},")
            f.write(f"{prd},")
            f.write(f"{obs},")
            f.write(f"{p}\n")

    predictive_fn = f"{output_path}/predictive.csv"
    with open(predictive_fn, "w") as f:
        f.write("clx,")
        f.write("pred_sr,")
        f.write("prob\n")
        for clx, prds in pred.items():
            for prd, p in sorted(prds.items(), key=lambda it: it[1], reverse=True):
                f.write(f"{clx},")
                f.write(f"{prd},")
                f.write(f"{p}\n")


if __name__ == "__main__":
    main()
