import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def parity(df, model):
    true = list(map(float, df["true"].tolist()))
    pred = list(map(float, df[model].tolist()))
    
    i = 0
    while i < len(true):
        if pred[i] < 0 or pred[i] > 1000:
            true.pop(i)
            pred.pop(i)
        else:
            i += 1

    df = pd.DataFrame({"true": true, "pred": pred})
    
    min = 100
    max = 650
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "#ffffff"), (0.5, "#3333aa"), (1, "#000099")])
    df.plot.hexbin(x="true", y="pred", gridsize=40, extent=[min, max, min, max], cmap=cmap)

    z = np.polyfit(true, pred, 1)
    p = np.poly1d(z)
    plt.plot(true, p(true), color="red", linewidth="2", linestyle="--")

    plt.xlabel("True Temps")
    plt.ylabel("Predicted Temps")
    plt.plot([min, max], [min, max], color="red")
    plt.savefig(f"{model}_hexbin.png")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.abspath("."), sys.argv[1]))
    model = sys.argv[2]
    parity(df, model)