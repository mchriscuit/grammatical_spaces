import pandas as pd
import re
import json
import argparse


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
            ARGUMENT PARSER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ifp", nargs="+", type=str, required=True, help="")
    parser.add_argument("-ofp", nargs="?", type=str, required=True, help="")
    args = parser.parse_args()
    return args


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
            MAIN FUNCTION CALL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():
    ## Retrieve the arguments from the commandline
    args = parse_args()

    ## Get the filenames of the predictive posteriors
    ifps = args.ifp

    ## Loop through each of the files; in the directory outside of the folder
    ## containing the predictive posterior file, we expect there to be a json
    ## with title ``expected`` with labels corresponding to the expected and
    ## optionally the alternative outputs for the test items. If no alternatives
    ## are given, we do not renormalize; if alternatives are given, then we
    ## renormalized the probabilities based on the sum of the two options
    dfs = []
    for fp in ifps:
        ## Read in the expected file and predictive files
        pt = f"{re.sub(r'(?<=/)lm.*', '', fp)}{'expected.json'}"
        with open(pt) as fpt, open(fp) as ffp:
            jex = json.load(fpt)
            jdf = json.load(ffp)

        ## Clean the dataframe to eliminate padding
        jdf = {
            cx: {re.sub("#", "", f): p for f, p in i.items()} for cx, i in jdf.items()
        }

        ## Create a dataframe structure to follow the following organization:
        ## (1) the outermost dictionary corresponds to the lexical item (cx)
        ## (2) the inner dictionary corresponds to the expected forms and
        ##     their posterior predictive
        check = list(jex.values())[0]
        if "int_sr" in check and "alt_sr" in check:
            df = {"cx": [], "trial": [], "int": [], "alt": [], "prb": []}
            for cx, info in jex.items():
                intd = info["int_sr"]
                altd = info["alt_sr"]
                trial = info["trial"]
                pint = jdf[cx].get(intd, 0)
                palt = jdf[cx].get(altd, 0)
                pnrm = pint / (pint + palt)
                df["cx"].append(cx)
                df["trial"].append(trial)
                df["int"].append(intd)
                df["alt"].append(altd)
                df["prb"].append(pnrm)
        elif "int_sr" in check:
            df = {"cx": [], "trial": [], "int": [], "prb": []}
            for cx, info in jex.items():
                intd = info["int_sr"]
                trial = info["trial"]
                pint = jdf[cx].get(intd, 0)
                df["cx"].append(cx)
                df["trial"].append(trial)
                df["int"].append(intd)
                df["prb"].append(pint)

        ## Transform to dataframe
        df = pd.DataFrame(df)

        ## Add language column to dataframe
        lg = re.sub(r"outputs/(.*?)/lm.*", r"\1", fp)
        df["lng"] = lg

        ## Append dataframe to list
        dfs.append(df)

    ## Generate a single dataframe
    dfs = pd.concat(dfs)

    ## Save dataframe to file
    ofp = args.ofp
    dfs.to_csv(ofp)


if __name__ == "__main__":
    main()
