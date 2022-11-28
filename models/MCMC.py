import numpy as np
import re
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from optim.Lexicon import Lexicon
from optim.Phonology import SPE, OT
from optim.Grammar import Grammar

""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                MCMC SAMPLER DEFINITION
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


class MCMC:

    """========== STATIC METHODS ====================================="""

    @staticmethod
    def acceptance(posterior_old, posterior_new):
        """Returns a boolean corresponding whether to accept or reject
        the new sample
        """
        if posterior_new > posterior_old:
            return True
        elif posterior_new == 0 and posterior_old == 0:
            return True
        else:
            return np.random.uniform(0, 1) < (posterior_new / posterior_old)

    @staticmethod
    def gibbs_sampler(G, anneal: float, gs_iterations: int, mh_iterations: int):
        """Performs the Gibbs sampling algorithm for the Grammar object"""

        ## Initialize RNG generator
        rng = np.random.default_rng()

        ## Initialize acceptances for each lexeme
        lx_acceptances = {lx: [] for lx in G.L.lxs()}

        ## Initialize list to store grammars at the end of each iteration
        sampled_grammars = []
        for gs_iteration in tqdm(range(gs_iterations)):

            ## Compute annealing exponent
            power = min(np.log(gs_iteration + 2) / anneal, 1)

            ## Loop through each contextual UR
            for clx in G.clxs():

                ## Retrieve the prior and likelihood of the old UR
                cxt_ur_old = G.L.cxt_ur(clx)
                cxt_pr_old = G.L.cxt_pr(clx)
                likelihood_old = G.compute_likelihood(clx, G.levenshtein)

                ## Loop for mh_iterations
                for mh_iteration in range(mh_iterations):
                    
                    ## Sample a new contextual UR
                    cxt_ur_new, cxt_pr_new = G.L.sample_cxt_ur(clx)
                    likelihood_new = G.compute_likelihood(clx, G.levenshtein)

                    ## Calculate the transition probability of the hypotheses
                    tp_old = np.prod(
                        [G.L.compute_tp(n, o) for n, o in zip(cxt_ur_new, cxt_ur_old)]
                    )
                    tp_new = np.prod(
                        [G.L.compute_tp(o, n) for o, n in zip(cxt_ur_old, cxt_ur_new)]
                    )

                    ## Accept or reject the sample
                    post_old = tp_new * (likelihood_old * np.prod(cxt_pr_old)) ** power
                    post_new = tp_old * (likelihood_new * np.prod(cxt_pr_new)) ** power
                    accepted = MCMC.acceptance(post_old, post_new)

                    ## If we do not accept, revert to the old UR hypothesis
                    if not accepted:
                        for lx, cxt_u_old, cxt_p_old in zip(clx, cxt_ur_old, cxt_pr_old): 
                            G.L.set_ur(lx, clx, cxt_u_old, cxt_p_old)

                    ## Otherwise, update the new hypothesis
                    else:
                        cxt_ur_old = G.L.cxt_ur(clx)
                        cxt_pr_old = G.L.cxt_pr(clx)
                        likelihood_old = likelihood_new

            ## Loop through each prototype UR
            for lx in G.lxs():

                ## Retrieve the old prototype UR and prior
                pro_ur_old = G.L.pro_ur(lx)
                pr_old = deepcopy(G.L.lx_pr(lx))
                prod_pr_old = np.prod(list(pr_old.values()))

                ## Loop for mh_iterations
                for mh_iteration in range(mh_iterations):

                    ## Sample a new prototype underlying form
                    G.L.sample_pro_ur(lx)
                    pro_ur_new = G.L.pro_ur(lx)
                    pr_new = G.L.lx_pr(lx)
                    prod_pr_new = np.prod(list(pr_new.values()))

                    ## Calculate the transition probability of the hypotheses
                    tp_old = G.L.compute_tp(pro_ur_new, pro_ur_old)
                    tp_new = G.L.compute_tp(pro_ur_old, pro_ur_new)

                    ## Accept or reject the sample
                    post_old = (prod_pr_old ** power) * tp_new
                    post_new = (prod_pr_new ** power) * tp_old
                    accepted = MCMC.acceptance(post_old, post_new)

                    ## If we do not accept, revert to the old UR hypothesis
                    if not accepted:
                        G.L.set_pro_ur(lx, pro_ur_old, pr_old)

                    ## Otherwise, update the new hypothesis
                    else:
                        pro_ur_old = G.L.pro_ur(lx)
                        pr_old = deepcopy(G.L.lx_pr(lx))
                        prod_pr_old = np.prod(list(pr_old.values()))

            ## Loop through each mapping hypothesis
            nmhyps = G.M.nmhyps()
            pmhyps = np.ones(nmhyps)
            for mhyp_idx in range(nmhyps):

                ## Update the mapping hypothesis internally
                G.M.update_mhyp_idx(mhyp_idx)

                ## Calculate the likelihood and prior of the mapping hypothesis
                likelihood = np.prod(
                    [
                        G.compute_likelihood(clx, G.levenshtein)
                        for clx in G.clxs()
                    ]
                )
                prior = G.M.compute_prior()
                pmhyps[mhyp_idx] = (likelihood * prior) ** power

            ## Sample a mapping hypothesis based on the unnormalized posterior
            if np.sum(pmhyps) == 0:
                pmhyps = np.ones(pmhyps.shape)
            pmhyps = pmhyps / np.sum(pmhyps)
            sampled_mhyp_idx = rng.choice(nmhyps, p=pmhyps)

            ## Update the mapping hypothesis with the sampled mapping hypothesis
            G.M.update_mhyp_idx(sampled_mhyp_idx)

            ## Append to sample
            sampled_grammars.append(deepcopy(G))

        return sampled_grammars, lx_acceptances

    @staticmethod
    def burn_in(sampled_grammars: list, burn_in=2, steps=20):
        """Burns in the sampled grammars by the specified amount"""
        nsampled_grammars = len(sampled_grammars)
        burn_in_idx = int(nsampled_grammars / burn_in)
        return sampled_grammars[burn_in_idx::steps]

    @staticmethod
    def posterior_predictive(burned_in: list):
        """Calculates the posterior predictive probability for each output
        given the burned-in sample posterior
        """
        post = {}
        pred = defaultdict(dict)

        ## Populate dictionaries
        for grammar in tqdm(burned_in):
            clxs, mnames, urs, pred_srs, obs_srs = grammar.export()
            clxs = [":".join(cxs) for cxs in clxs]
            for clx, pred_sr in zip(clxs, pred_srs):
                pred[clx][pred_sr] = pred[clx].get(pred_sr, 0) + 1
            clxs = ".".join(clxs)
            mnames = ".".join(mnames)
            urs = ".".join(urs)
            pred_srs = ".".join(pred_srs)
            obs_srs = ".".join(obs_srs)
            fgrammar = (clxs, mnames, urs, pred_srs, obs_srs)
            post[fgrammar] = post.get(fgrammar, 0) + 1

        ## Normalize counts to get empirical probabilities
        post = {fg: count / sum(post.values()) for fg, count in post.items()}
        pred = {
            clx: {
                sr: count / sum(pred[clx].values())
                for sr, count in pred[clx].items()
            }
            for clx in pred
        }
        return post, pred
