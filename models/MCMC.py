import numpy as np
import re
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from models.Inventory import Inventory
from models.Lexicon import Lexicon
from models.Phonology import SPE, OT
from models.Grammar import Grammar

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
    def gibbs_sampler(G, gs_iterations: int, mh_iterations: int):
        """Performs the Gibbs sampling algorithm for the Grammar object"""

        ## Initialize list to store grammars at the end of each iteration
        sampled_grammars = []
        rng = np.random.default_rng()
        for gs_iteration in tqdm(range(gs_iterations)):

            ## Loop through each lexeme and get the current UR hypotheses
            ## Calculate the likelihood of the data given the lexemes
            for lx in G.L.lxs():
                for mh_iteration in range(mh_iterations):

                    ## Retrieve the old UR hypotheses
                    ur_old, aln_old, p_trans_ur_old = G.L.get_hyp(lx)
                    prior_old = G.L.calculate_pr(ur_old, aln_old)
                    likelihood_old = G.compute_likelihoods(lx, G.levenshtein)

                    ## Sample a new UR hypothesis
                    ur_new, aln_new, p_trans_ur_new = G.L.sample_ur(lx, inplace=True)
                    prior_new = G.L.calculate_pr(ur_new, aln_new)
                    likelihood_new = G.compute_likelihoods(lx, G.levenshtein)

                    ## Accept or reject the sample
                    posterior_old = likelihood_old * prior_old * p_trans_ur_new
                    posterior_new = likelihood_new * prior_new * p_trans_ur_old
                    accepted = MCMC.acceptance(posterior_old, posterior_new)

                    ## If we do not accept, revert to the old UR hypothesis
                    if not accepted:
                        G.L.set_ur(lx, ur_old, aln_old, p_trans_ur_old)

            ## Loop through each mapping hypothesis
            ## Calculate the likelihood of the data given each mapping
            pmhyps = []
            nmhyps = G.M.nmhyps()
            for mhyp_idx in range(nmhyps):
                G.M.update_mhyp_idx(mhyp_idx)
                likelihood = np.prod(
                    [G.compute_likelihoods(lx, G.levenshtein) for lx in G.L.lxs()]
                )
                prior = G.M.compute_prior()
                pmhyps.append(likelihood * prior)
            pmhyps = np.array(pmhyps)
            if np.sum(pmhyps) == 0:
                pmhyps = np.ones(pmhyps.shape)
            pmhyps = pmhyps / np.sum(pmhyps)
            sampled_mhyp_idx = rng.choice(nmhyps, p=pmhyps)
            G.M.update_mhyp_idx(sampled_mhyp_idx)

            ## Append to sample
            sampled_grammars.append(deepcopy(G))

        return sampled_grammars

    @staticmethod
    def burn_in(sampled_grammars: list, burn_in=2, steps=20):
        """Burns in the sampled grammars by the specified amounts"""
        nsampled_grammars = len(sampled_grammars)
        burn_in_idx = int(nsampled_grammars / burn_in)
        return sampled_grammars[burn_in_idx::steps]

    @staticmethod
    def posterior_predictive(burned_in_sampled_grammars: list):
        """Calculates the posterior predictive probability for each output
        given the burned-in sample posterior
        """
        predictive = defaultdict(dict)
        for burned_in_sampled_grammar in tqdm(burned_in_sampled_grammars):
            clxs, mnames, urs, pred_srs, obs_srs = burned_in_sampled_grammar.export()
            for clx, pred_sr in zip(clxs, pred_srs):
                predictive[clx][pred_sr] = predictive[clx].get(pred_sr, 0) + 1
        predictive = {
            clx: {
                pred: count / sum(predictive[clx].values())
                for pred, count in predictive[clx].items()
            }
            for clx in predictive
        }
        return predictive
