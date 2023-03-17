import multiprocessing as mp
import numpy as np

from utils.process import initialize
from optim.Grammar import Grammar


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                MAIN FUNCTION CALL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():

    ## Load and initialize ``Grammar`` object
    G = initialize()

    assert False

    """ ==================== RUN MODEL ===================================="""
    gs_iterations = params["MCMC"]["gs_iterations"]
    mh_iterations = params["MCMC"]["mh_iterations"]
    anneal = params["MCMC"]["anneal"]
    sg = MCMC.gibbs_sampler(G, anneal, gs_iterations, mh_iterations)

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
    output_path += f"-lambda{lm}-psi{psi}-phi{phi}-alpha{alpha}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
