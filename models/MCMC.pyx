## Basic Imports
import numpy as np
import re
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

## Multiprocessing Imports
from functools import partial
from pathos.multiprocessing import ProcessPool as Pool

## Self-Defined Imports
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

        ## If the new sample has a higher joint probability than the old
        ## sample, automatically accept the new sample
        if posterior_new > posterior_old:
            return True

        ## If both the new and old sample have a joint probability of 0,
        ## automatically accept the new sample
        elif posterior_new == 0 and posterior_old == 0:
            return True

        ## Otherwise, sample based on the ratio of the two joint probabilities
        else:
            return np.random.uniform(0, 1) < (posterior_new / posterior_old)

    @staticmethod
    def gibbs_sampler(
        G, anneal: float, gs_iterations: int, mh_iterations: int, n_cores: int = 2
    ):
        """Performs the Gibbs sampling algorithm for the Grammar object"""

        ## Initialize RNG generator
        rng = np.random.default_rng()

        ## Initialize rule cache
        nmhyps = G.M.nmhyps()
        mhyp_idxs = list(range(nmhyps))

        ## Pre-generate multiprocessesing splits on the rule hypotheses
        # mhyp_idxs_split = np.array_split(mhyp_idxs, n_cores)

        ## Generate the pools
        # pool = Pool(n_cores)

        ## Initialize list to store grammars at the end of each iteration
        sampled_grammars = []
        for gs_iteration in tqdm(range(gs_iterations)):
            ## Compute annealing exponent
            power = min(np.log(gs_iteration + 2) / anneal, 1)

            ## Loop through each contextual UR
            for cxt in G.cxts():
                ## Retrieve the prior and likelihood of the old UR
                cxt_ur_old, str_ur_old, cxt_pr_old, prd_pr_old = G.L.get_cxt_info(cxt)
                likelihood_old = G.compute_cxt_likelihoods(
                    [str_ur_old], [cxt], G.levenshtein
                )

                ## Loop for mh_iterations
                for mh_iteration in range(mh_iterations):
                    ## Sample a new contextual UR
                    cxt_ur_new, str_ur_new, cxt_pr_new, prd_pr_new = G.L.sample_cxt_ur(
                        cxt
                    )
                    likelihood_new = G.compute_cxt_likelihoods(
                        [str_ur_new], [cxt], G.levenshtein
                    )

                    ## Calculate the transition probabilities
                    tp_old_new, tp_new_old = G.L.compute_clx_cxt_tp(
                        cxt_ur_old, cxt_ur_new
                    )

                    ## Accept or reject the sample
                    post_old = tp_old_new * (likelihood_old * prd_pr_old) ** power
                    post_new = tp_new_old * (likelihood_new * prd_pr_new) ** power
                    accepted = MCMC.acceptance(post_old, post_new)

                    ## If it is accepted, update the grammar object
                    if accepted:
                        G.L.set_ur(cxt_ur_new, cxt_pr_new)
                        (
                            cxt_ur_old,
                            str_ur_old,
                            cxt_pr_old,
                            prd_pr_old,
                        ) = G.L.get_cxt_info(cxt)
                        likelihood_old = likelihood_new

            ## Loop through each lexeme
            for lx in G.lxs():
                ## Retrieve the contexts the lexeme is found in
                cxts = G.L.lx2cxts(lx)

                ## Retrieve the old prototype UR and prior
                pro_ur_old = G.L.pro_ur(lx)
                str_urs_old = [G.L.str_cxt_ur(cxt) for cxt in cxts]
                prd_prs_old = np.prod([G.L.prd_cxt_pr(cxt) for cxt in cxts])

                ## Calculate the likelihood of the data
                likelihood_old = G.compute_cxt_likelihoods(
                    str_urs_old, cxts, G.levenshtein
                )

                ## Loop for mh_iterations
                for mh_iteration in range(mh_iterations):
                    ## Sample a new prototype underlying form and prior
                    (
                        pro_ur_new,
                        ur_new,
                        pr_new,
                        str_urs_new,
                        prd_prs_new,
                    ) = G.L.sample_pro_ur(lx, cxts)

                    ## Calculate the likelihood of the data
                    likelihood_new = G.compute_cxt_likelihoods(
                        str_urs_new, cxts, G.levenshtein
                    )

                    ## Calculate the transition probability of the hypotheses
                    tp_old_new, tp_new_old = G.L.compute_pro_tp(pro_ur_new, pro_ur_old)

                    ## Accept or reject the sample
                    post_old = tp_old_new * (likelihood_old * prd_prs_old) ** power
                    post_new = tp_new_old * (likelihood_new * prd_prs_new) ** power
                    accepted = MCMC.acceptance(post_old, post_new)

                    ## If it is accepted, update the grammar object
                    if accepted:
                        G.L.set_ur(ur_new, pr_new)
                        pro_ur_old = pro_ur_new
                        str_urs_old = str_urs_new
                        prd_prs_old = prd_prs_new
                        likelihood_old = likelihood_new

            ## Calculate the weighted posteriors of each mapping hypothesis
            # likelihood = np.concatenate(
            #     pool.map(
            #         partial(rule_inference, G=G),
            #         mhyp_idxs_split,
            #     )
            # )
            likelihood = G.compute_all_likelihoods(G.levenshtein, mhyp_idxs)
            prior = 1
            pmhyps = (likelihood * prior) ** power

            ## Sample a mapping hypothesis based on the unnormalized posterior
            if np.sum(pmhyps) == 0:
                pmhyps = np.ones(pmhyps.shape)
            pmhyps = pmhyps / np.sum(pmhyps)
            sampled_mhyp_idx = rng.choice(nmhyps, p=pmhyps)

            ## Update the mapping hypothesis with the sampled mapping hypothesis
            G.M.update_mhyp_idx(sampled_mhyp_idx)

            ## Append to sample
            sampled_grammars.append(G.export())

        ## Close and join the pool
        # pool.close()
        # pool.join()

        return sampled_grammars

    @staticmethod
    def burn_in(sampled_grammars: list, burn_in: int = 2, steps: int = 20):
        """Burns in the sampled grammars by the specified amount"""

        ## Get the number of sampled grammars
        n_samples = len(sampled_grammars)

        ## Get the number of samples to burn in
        idx = int(n_samples / burn_in)

        return sampled_grammars[idx::steps]

    @staticmethod
    def posterior_predictive(burned_in: list):
        """Calculates the posterior predictive probability for each output
        given the burned-in sample posterior
        """
        post = {}
        pred = defaultdict(dict)

        ## Populate dictionaries
        for grammar in tqdm(burned_in):
            ## Retrieve the information from the grammar
            clxs, mnames, urs, pred_srs, obs_srs = grammar

            ## Count each predicted SR for each clx
            for clx, pred_sr in zip(clxs, pred_srs):
                pred[clx][pred_sr] = pred[clx].get(pred_sr, 0) + 1

            ## Format the strings of all the exported information
            clxs = ".".join(clxs)
            mnames = ".".join(mnames)
            urs = ".".join(urs)
            pred_srs = ".".join(pred_srs)
            obs_srs = ".".join(obs_srs)

            ## Count the number of each grammar
            fgrammar = (clxs, mnames, urs, pred_srs, obs_srs)
            post[fgrammar] = post.get(fgrammar, 0) + 1

        ## Normalize counts to get empirical probabilities
        post = {fg: count / sum(post.values()) for fg, count in post.items()}
        pred = {
            clx: {
                sr: count / sum(pred[clx].values()) for sr, count in pred[clx].items()
            }
            for clx in pred
        }

        return post, pred


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
                MULTIPROCESSING HELPER FUNCTIONS
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """

def rule_inference(G, mhyp_idxs=[]):
    return G.compute_all_likelihoods(G.levenshtein, mhyp_idxs)