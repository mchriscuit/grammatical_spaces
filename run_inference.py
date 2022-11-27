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
    lx_params = lexicon["params"]
    lxs, xclxs, clxs, srs, nobs = load_surface_forms(surface_forms_fn)
    L = Lexicon(lxs, xclxs, lx_params, tokens, feats, configs)

    ## Initialize UR hypothesis
    lexicon_fn = lexicon["fn"]
    if lexicon_fn:
        L.initialize_urs(load_lexicon(lexicon_fn))
    else:
        L.initialize_urs()

    ## Load and return grammar object
    lm = surface_forms["lambda"]
    G = Grammar(clxs, srs, nobs, lm, L, M)
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
    init_lxs, init_clxs, init_urs = zip(*lexicon)
    init_lxs = list(init_lxs)
    init_clxs = [[tuple(x.split(":")) for x in cx.split(".")] for cx in init_clxs]
    init_urs = [ur.split(".") for ur in init_urs]
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
    clxs = [tuple(clx.split(":")) for clx in clxs]
    lxs = sorted(set(lx for clx in clxs for lx in clx))
    xclxs = [[clx for clx in clxs if lx in clx] for lx in lxs]
    xclxs = [[tuple(["PROTO"])] + cxs for cxs in xclxs]
    srs = list(srs)
    nobs = [int(nob) for nob in nobs]
    print(clxs, srs)
    return lxs, xclxs, clxs, srs, nobs


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

    ## Generate new directory
    output_path = f"./output/"
    output_path += f"{re.sub('.csv', '', surface_forms['fn'])}"
    output_path += f"-gs{gs_iterations}-mh{mh_iterations}"
    output_path += f"-lm{lm}-psi{psi}-phi{phi}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## Write to file
    acceptances_fn = f"{output_path}/acceptances.csv"
    with open(acceptances_fn, "w") as f:
        f.write("lx,")
        f.write("gs,")
        f.write("mh,")
        f.write("mhyp,")
        f.write("ur_old,")
        f.write("post_old,")
        f.write("ur_new,")
        f.write("post_new,")
        f.write("accepted\n")
        for lx, acs in lx_as.items():
            for gsi, ac in enumerate(acs):
                for mhi, (m, uo, po, un, pn, ad) in enumerate(ac):
                    f.write(f"{lx},")
                    f.write(f"{gsi},")
                    f.write(f"{mhi},")
                    f.write(f"{m},")
                    f.write(f"{'-'.join(uo)},")
                    f.write(f"{po},")
                    f.write(f"{'-'.join(un)},")
                    f.write(f"{pn},")
                    f.write(f"{ad}\n")

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
