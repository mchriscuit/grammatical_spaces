import numpy as np
import copy as cp
import json

from tqdm import tqdm
from collections import defaultdict
from utils.process import initialize
from models.Grammar import Grammar
from models.Phonology import SPE
from models.Lexicon import Lexicon


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

    # print(o)
    # print(n)
    # print(n / o)
    # print(h)
    # print(z)
    # print(s)
    # print("")

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
    midx = np.arange(G.M.nmhs)

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
                olk = np.tile(np.diagonal(G.likelihood(oexp, G.srs)), (len(x), 1))
                nlk = np.tile(np.diagonal(G.likelihood(nexp, G.srs)), (len(x), 1))

                ## Create a matrix containing the likelihoods
                ol = np.ones(olk.shape)
                nl = np.ones(nlk.shape)

                ## Update the relevant slots based on the index
                ol[pid] = olk[pid]
                nl[pid] = nlk[pid]

                ## Calculate the weighted posteriors
                oj = ontp * (ol.prod(axis=1) * oprpro[x] * oprcxt[x].prod(axis=1)) ** pw
                nj = notp * (nl.prod(axis=1) * nprpro[x] * nprcxt[x].prod(axis=1)) ** pw

                ## Initialize and fill acceptance vector
                a = np.full(G.L.nlxs, False)
                a[x] = acceptance(oj, nj)

                # print("opro", opro)
                # print("ocxt", ocxt)
                # print("oidy", idy)
                # print("oprpro", oprpro[x])
                # print("oprcxt", oprcxt[x])
                # print("ol", ol)
                # print("ontp", ontp)
                # print("oj", oj)

                # print("npro", npro)
                # print("ncxt", ncxt)
                # print("nidy", idy)
                # print("nprpro", nprpro[x])
                # print("nprcxt", nprcxt[x])
                # print("nl", nl)
                # print("notp", notp)
                # print("nj", nj)

                # print(a)
                # print("")

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
            olk = G.likelihood(oexp, G.srs)
            nlk = G.likelihood(nexp, G.srs)

            ## Create a matrix containing the likelihoods
            ol = np.ones(olk[:, 0].shape)
            nl = np.ones(nlk[:, 0].shape)

            ## Update the relevant slots based on the index
            ol[G.oid] = olk[G.oid, 0]
            nl[G.oid] = nlk[G.oid, 0]

            # print("====================================")
            # print("OLD")
            # print("====================================")
            # print("ocxt", ocxt)
            # print("oidy", oidy)
            # print("oexp", oexp.squeeze())
            # print("srs", G.srs)
            # print(oprcxt)
            # print(ol)
            # print(ontp)

            # print("====================================")
            # print("NEW")
            # print("====================================")
            # print("ncxt", ncxt)
            # print("nidy", nidy)
            # print("nexp", nexp.squeeze())
            # print("srs", G.srs)
            # print(nprcxt)
            # print(nl)
            # print(notp)
            # print("")

            # assert False

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
        exp = G.M.apply(G.L.join(G.L.cxt), midx).T
        lik = G.likelihood(exp, G.srs)[:, G.oid].prod(axis=1)
        lik = lik / lik.sum()

        ## Sample a new mapping hypothesis based on the likelihoods
        G.M.mhid = rng.choice(np.arange(lik.size), p=lik)

        ## After the burn-in period, for every xth iteration,
        ## save the hypotheses and their predictions
        if gs >= burnIdx and gs % skIters == 0:
            smp = {
                "lxs": G.L.lxs,
                "pro": G.L.pro,
                "cxs": G.cxs,
                "cxt": G.L.join(G.L.cxt, ""),
                "idy": G.L.idy[G.L.cid],
                "pds": exp[G.M.mhid, :].squeeze(),
                "mhy": G.M.mhys[G.M.mhid],
                "obs": G.srs,
            }
            samples.append(cp.deepcopy(smp))

    return samples


def posterior(B: list):
    """Calculates the posterior and posterior predictive given a list of sampled grammars"""
    post = {}
    pred = defaultdict(dict)

    ## Populate dictionaries
    for bmp in tqdm(B):

        ## Retrieve the information from the grammar
        cxs, pds = bmp["cxs"], bmp["pds"]

        ## Count each predicted SR for each clx
        for cx, pd in zip(cxs, pds):
            fcx = "-".join(cx)
            pred[fcx][pd] = pred[fcx].get(pd, 0) + 1

    ## Normalize counts to get empirical probabilities
    pred = {
        fcx: {pd: cnt / sum(pred[fcx].values()) for pd, cnt in pred[fcx].items()}
        for fcx in pred
    }
    return pred


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                MAIN FUNCTION CALL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():

    ## Load and initialize ``Grammar`` object
    G, C = initialize()

    ## Run the MCMC algorithm
    B = sample(G, **C)

    ## Calculate the posteriors and posterior predictive
    pred = posterior(B)
    print(pred)

    ## Save to file
    with open("test.json", "w") as f:
        json.dump(pred, f, indent=4)
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
