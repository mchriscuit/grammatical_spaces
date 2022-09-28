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
    def gibbs_sampler(G, iterations: int):
        """Performs the Gibbs sampling algorithm for the Grammar object"""
        sampled_grammars = []
        rng = np.random.default_rng()
        for iteration in tqdm(range(iterations)):

            ## Loop through each lexeme and get the current UR hypotheses
            ## Calculate the likelihood of the data given the lexemes
            for lx in G.L.lxs():
                plhyps = []
                nlhyps = G.L.nlhyps(lx)
                for lhyp_idx in range(nlhyps):
                    G.L.update_lhyp_idx(lx, lhyp_idx)
                    likelihood = G.compute_likelihood(lx)
                    prior = G.L.compute_prior(lx)
                    plhyps.append(likelihood * prior)
                plhyps = np.array(plhyps)
                if np.sum(plhyps) == 0:
                    plhyps = np.ones(plhyps.shape)
                plhyps = plhyps / np.sum(plhyps)
                sampled_lhyp_idx = rng.choice(nlhyps, p=plhyps)
                G.L.update_lhyp_idx(lx, sampled_lhyp_idx)

            ## Loop through the rule hypotheses
            ## Calculate the likelihood of the data given each rule ordering
            pmhyps = []
            nmhyps = G.M.nmhyps()
            for mhyp_idx in range(nmhyps):
                G.M.update_mhyp_idx(mhyp_idx)
                likelihood = np.prod([G.compute_likelihood(lx) for lx in G.L.lxs()])
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
            clxs, mnames, urs, pred, srs = burned_in_sampled_grammar.export()
            for clx, sr in zip(clxs, srs):
                predictive[clx][sr] = predictive[clx].get(sr, 0) + 1
        predictive = {
            clx: {
                pred: count / sum(predictive[clx].values())
                for pred, count in predictive[clx].items()
            }
            for clx in predictive
        }
        return predictive
