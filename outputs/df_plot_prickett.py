import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
            ARGUMENT PARSER
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ifp", nargs="?", type=str, required=True, help="")
    parser.add_argument("-ofp", nargs="?", type=str, required=True, help="")
    args = parser.parse_args()
    return args


""" *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
            MAIN FUNCTION CALL
=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* """


def main():
    ## Retrieve the arguments from the commandline
    args = parse_args()

    ## Get the filenames of the dataframe
    ifps = args.ifp

    ## Read in the csv as a pandas dataframe
    df = pd.read_csv(ifps)

    ## Rename cell names
    df["lng"] = [re.sub(".*-", "", l) for l in df["lng"]]

    ## Order the dataframe
    lg = ["b", "f", "cb", "cf"]
    tr = ["faithful", "deleting", "palatalizing", "interacting"]
    df = df.astype({"lng": pd.CategoricalDtype(lg, ordered=True)})
    df = df.astype({"trial": pd.CategoricalDtype(tr, ordered=True)})
    df = df.sort_values(["lng", "trial"])

    ## Set plot parameters
    plt.rcParams["text.usetex"] = True
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["legend.handlelength"] = 1
    plt.rcParams["legend.handleheight"] = 1.125

    ## Generate plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ## Create grid
    ax.grid(axis="y", color="grey", alpha=0.5, linewidth=0.5, zorder=0)

    ## Add whitespace
    fig.subplots_adjust(bottom=0.25, wspace=0.1)

    ## Create X positions to plot bars
    x = np.arange(len(tr))
    w = 0.2
    p = 2 * (-w)

    ## Generate bars
    colors = ["dimgrey", "darkgrey", "black", "whitesmoke"]
    for color, level in zip(colors, lg):
        ax.bar(
            x=x + p,
            height=df[df["lng"] == level]["nprb"],
            width=w,
            label=level,
            capsize=1.5,
            color=color,
            edgecolor="black",
            linewidth=0.2,
            error_kw=dict(lw=0.5, capsize=3, capthick=0.5),
            zorder=3,
        )
        plt.xticks(x - 0.1, ["Faithful", "Deleting", "Palatalizing", "Interacting"])
        p += w

    ax.tick_params(axis="x", direction="inout", labelsize=14, length=0)
    ax.tick_params(axis="y", direction="inout", labelsize=14)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Trial Type", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)

    ## Set legend and labels
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(
        handles=handles,
        labels=["Bleeding", "Feeding", "Counter-bleeding", "Counter-feeding"],
        fontsize=12,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=4,
        fancybox=False,
        columnspacing=0.8,
        handletextpad=0.5,
    ).get_frame().set_edgecolor("black")

    ## Save the figure
    plt.tight_layout
    plt.savefig(fname=args.ofp, format="pdf", transparent=True)


if __name__ == "__main__":
    main()
