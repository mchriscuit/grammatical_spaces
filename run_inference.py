import numpy as np
import copy as cp
import logging
import json
import re
import os
from ast import literal_eval
from collections import defaultdict
from tqdm import tqdm
from utils.process import load_parameters, process_parameters
from models.Grammar import Grammar


def acceptance(o: np.ndarray, n: np.ndarray):
    """Returns a boolean corresponding whether to accept or reject the new sample"""

    ## If the new sample has a higher joint probability than the old
    ## sample, automatically accept the new sample
    h = o < n

    ## If both the new and old sample have a joint probability of 0,
    ## automatically accept the new sample
    z = (o == 0) & (n == 0)

    ## Otherwise, sample a boolean corrresponding whetherr to accept
    ## or reject each parameter based on the ratio of the two joint probabilities
    s = np.random.uniform(0, 1, size=o.shape) < (n / o)

    return np.logical_or.reduce((h, z, s))


def sample(
    G: Grammar,
    gsIters: float,
    mhIters: float,
    skIters: float,
    anConst: float,
    biConst: float,
):

    ## Initialize RNG generator
    rng = np.random.default_rng()

    ## Initialize rule hypothesis indices
    midx = np.arange(G.M.nhs)

    ## Get the burn-in value, which tells us when to start taking samples
    burnIdx = gsIters // biConst

    ## Initialize a list of samples
    samples = []

    ## Perform the Gibbs sampling algorithm for gsIters iterations
    for gs in tqdm(range(gsIters)):

        ## Compute annealing exponent by taking the log of the current iteration
        ## divided by some constant. This should be set so that the exponent reaches
        ## a value of 1 around halfway into sampling (where burn-in takes place)
        pw = min(np.log(gs + 2) / anConst, 1)

        """ (1) Prototype Underlying Forms =================="""

        ## Loop through each independent group of lexemes
        for x in G.gid:

            ## Get the ids for each cxt a lexeme is associated with
            pid = G.pid[x]

            ## Sample prototype underlying forms
            for _ in range(mhIters):

                ## Retrieve the identity parameters
                idy = G.L.idy

                ## Retrieve the current prototype and contextual URs
                opro, ocxt, oprpro, oprcxt = G.L.pro, G.L.cxt, G.L.prpro, G.L.prcxt

                ## Sample a new prototype UR given the current prototype UR
                npro, ncxt, nprpro, nprcxt = G.L.sample_pro(opro, ocxt, idy)

                ## Calculate the transition probabilities of the old to
                ## new prototype underlying forms and vice versa
                ontp, notp = G.L.tppro(opro[x], npro[x])

                ## Calculate the expected outputs
                oexp = G.M.apply(G.L.join(ocxt))
                nexp = G.M.apply(G.L.join(ncxt))

                ## Calculate the likelihood of each of the contexts
                olk = np.tile(G.likelihood(oexp, G.obs), (len(x), 1))
                nlk = np.tile(G.likelihood(nexp, G.obs), (len(x), 1))

                ## Create a matrix containing the likelihoods
                ol = np.ones(ocxt[x].shape)
                nl = np.ones(ncxt[x].shape)

                ## Update the relevant slots based on the index
                ol[pid] = olk[pid]
                nl[pid] = nlk[pid]

                ## Calculate the weighted posteriors
                oj = ontp * (ol.prod(axis=1) * oprpro[x] * oprcxt[x].prod(axis=1)) ** pw
                nj = notp * (nl.prod(axis=1) * nprpro[x] * nprcxt[x].prod(axis=1)) ** pw

                ## Initialize and fill acceptance vector
                a = np.full(G.L.nlxs, False)
                a[x] = acceptance(oj, nj)

                ## If accepted, update the underlying form hypotheses
                G.L.pro[a] = npro[a]
                G.L.cxt[a, :] = ncxt[a, :]
                G.L.prpro[a] = nprpro[a]
                G.L.prcxt[a, :] = nprcxt[a, :]

        """ (2) Contextual Underlying Forms ================="""

        ## Loop through each contextual UR
        for _ in range(mhIters):

            ## Retrieve the current prototype and contextual URs
            ocxt, oidy, oprcxt = G.L.cxt, G.L.idy, G.L.prcxt

            ## Sample a contextual underlying form given the current
            ## underlying form and identity values
            ncxt, nidy, nprcxt = G.L.sample_cxt(ocxt, oidy)

            ## Calculate the transition probabilities of the old to
            ## new prototype underlying forms and vice versa
            ontp, notp = G.L.tpcxt(ocxt, oidy, ncxt, nidy)

            ## Calculate the expected outputs
            oexp = G.M.apply(G.L.join(ocxt))
            nexp = G.M.apply(G.L.join(ncxt))

            ## Calculate the likelihood of each of the contexts
            olk = G.likelihood(oexp, G.obs)
            nlk = G.likelihood(nexp, G.obs)

            ## Create a matrix containing the likelihoods
            ol = np.ones(olk.shape)
            nl = np.ones(nlk.shape)

            ## Update the relevant slots based on the index
            ol[G.oid] = olk[G.oid]
            nl[G.oid] = nlk[G.oid]

            ## Calculate the weighted posteriors
            po = ontp.prod(axis=0) * (ol * oprcxt.prod(axis=0)) ** pw
            pn = notp.prod(axis=0) * (nl * nprcxt.prod(axis=0)) ** pw

            ## Initialize and fill acceptance vector
            a = acceptance(po, pn)

            ## If accepted, update the underlying form hypotheses
            G.L.cxt[:, a] = ncxt[:, a]
            G.L.idy[:, a] = nidy[:, a]
            G.L.prcxt[:, a] = nprcxt[:, a]

        """ (3) Rule hypotheses ============================="""

        ## Calculate the expected outputs and likelihood of every context
        ## for each mapping hypothesis
        exs = G.M.apply(G.L.join(G.L.cxt), midx).T
        lks = G.likelihood(exs, G.obs)[:, G.oid].prod(axis=1)
        lks = lks / lks.sum()

        ## Sample a new mapping hypothesis based on the likelihoods
        G.M.mhi = rng.choice(np.arange(lks.size), p=lks)

        ## After the burn-in period, for every xth iteration,
        ## save the hypotheses and their predictions
        if gs >= burnIdx and gs % skIters == 0:
            smp = {
                "lxs": G.lxs,
                "cxs": G.cxs,
                "mhy": G.M.mhs[G.M.mhi][0],
                "pro": G.L.pro,
                "cxt": G.L.join(G.L.cxt, ""),
                "idy": G.L.idy[G.L.cid],
                "exs": exs[G.M.mhi].squeeze(),
                "obs": G.obs,
            }
            samples.append(cp.deepcopy(smp))

    return samples


def predictive(S: list):
    """Calculates the posterior predictive given a list of sampled grammars"""
    pred = defaultdict(dict)

    ## Populate dictionaries
    for smp in tqdm(S):

        ## Retrieve the information from the grammar
        cxs, exs = smp["cxs"], smp["exs"]

        ## Count each predicted SR for each clx
        for cx, ex in zip(cxs, exs):
            fcx = "-".join(cx)
            pred[fcx][ex] = pred[fcx].get(ex, 0) + 1

    ## Normalize counts to get empirical probabilities
    pred = {
        fcx: {ex: ct / sum(pred[fcx].values()) for ex, ct in pred[fcx].items()}
        for fcx in pred
    }

    return pred


def posteriors(S: list, keys: list):
    """Calculates the posterior given a list of sampled grammars"""
    post = defaultdict(int)

    ## Populate dictionaries
    for smp in tqdm(S):

        ## Retrieve the grammatical information
        info = [list(smp[k]) for k in keys]
        post[repr(info)] = post[repr(info)] + 1

    ## Normalize counts to get posterior probabilities
    post = {g: ct / sum(post.values()) for g, ct in post.items()}

    return post


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
                MAIN FUNCTION CALL
*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():

    """====== (1) Load Parameter Data =============================================="""

    ## Load in parameters of the model
    grP, mcP = load_parameters()

    ## Process each parameter of the model
    fn = grP["filenames"]

    ## Process the parameters of the model
    d, m, i = process_parameters(**fn)
    params = grP["parameters"]

    ## Get the parameters of the model
    lm = params["conservativity"]
    ml = params["maxlength"]
    mb = params["maxbuffer"]
    th = params["prolength"]
    ps = params["urprior"]
    ph = params["urtrans"]
    al = params["dfprior"]
    bt = params["dftrans"]
    gs = mcP["gsIters"]
    mh = mcP["mhIters"]

    """====== (2) Process Logging and Output Files ================================="""

    ## Retrieve the filename of the dataset
    dn = re.sub(r"^.*/(.*?)\..*", r"\1", grP["filenames"]["obs"])

    ## Process the file directory for the logging file; generate a directory if there is
    ## not one found
    dl = "docs/logs"
    lpath = f"{dl}/{dn}/lm{lm}-gs{gs}-mh{mh}-ml{ml}-th{th}-ps{ps}-ph{ph}-al{al}-bt{bt}/"
    if not os.path.exists(lpath):
        os.makedirs(lpath)
    logging.basicConfig(filename=f"{lpath}runtime.log", level=logging.INFO)

    ## Process the file directory for the output file; generate a directory if there is
    ## not one found
    do = "outputs"
    opath = f"{do}/{dn}/lm{lm}-gs{gs}-mh{mh}-ml{ml}-th{th}-ps{ps}-ph{ph}-al{al}-bt{bt}/"
    if not os.path.exists(opath):
        os.makedirs(opath)

    """====== (3) Initialize Grammar Object ========================================"""

    ## Initialize the model given the processed parameters
    print("\nInitializing ``Grammar`` object...")
    G = Grammar(*d, m, i, params)
    print("``Grammar`` object has been initialized!")

    """====== (3) Run Sampling Algorithm ==========================================="""

    ## Run the MCMC algorithm
    print("\nRunning the MCMC algorithm...")
    S = sample(G, **mcP)
    print("MCMC algorithm has completed running!")

    """====== (3) Save Outputs to File ============================================="""

    ## Retrieve posterior predictive distribution and save to file
    print("\nSaving samples to file...")
    Pr = predictive(S)
    Pr = {
        fcx: {
            ex: p for ex, p in sorted(Pr[fcx].items(), reverse=True, key=lambda k: k[1])
        }
        for fcx in Pr
    }
    with open(f"{opath}pred.json", "w") as f:
        json.dump(Pr, f, indent=2)

    ## Retrieve posterior distribution and save to file
    keys = ["pro", "cxt", "idy", "mhy", "exs"]
    Po = posteriors(S, keys)
    Po = [
        [{k: v for k, v in zip(keys, literal_eval(g))}, p]
        for g, p in sorted(Po.items(), reverse=True, key=lambda k: k[1])
    ]
    with open(f"{opath}post.json", "w") as f:
        json.dump(Po, f, indent=2)
    print("Samples successfully saved to file!")


if __name__ == "__main__":
    main()
