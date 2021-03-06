import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error

PATHNAME = os.path.abspath(".") + "/"

csv_path = sys.argv[1]

def r2(df, y_true_name):
    scores = {}

    y_true = list(map(float, df.loc[df["name"] == y_true_name, "y_predict"].tolist()[0][1:-1].split()))
    print(df.columns)
    for index, row in df.iterrows():
        if index > 0:
            y_pred = list(map(float, row["y_predict"][1:-1].split()))
            scores[row["name"]] = r2_score(y_true, y_pred)

    names = list(scores.keys())
    values = list(scores.values())
    plt.bar(range(len(scores)), values, align='center', tick_label=names)
    plt.xticks(rotation=90)
    ax = plt.gca()
    ax.set_ylim([-1, 1])
    plt.xlabel("Regression Model")
    plt.ylabel("R² Score")
    plt.tight_layout()
    plt.gca().set_aspect(16/2)
    plt.savefig("r2.png")
    plt.show()

def ame(df, y_true_name):
    scores = {}

    y_true = list(map(float, df.loc[df["name"] == y_true_name, "y_predict"].tolist()[0][1:-1].split()))
    print(df.columns)
    for index, row in df.iterrows():
        if index > 0:
            y_pred = list(map(float, row["y_predict"][1:-1].split()))
            scores[row["name"]] = mean_absolute_error(y_true, y_pred)

    names = list(scores.keys())
    values = list(scores.values())
    plt.bar(range(len(scores)), values, align='center', tick_label=names)
    plt.xticks(rotation=90)
    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.xlabel("Regression Model")
    plt.ylabel("Absolute Mean Error")
    plt.tight_layout()
    plt.gca().set_aspect(16/100)
    plt.savefig("ame.png")
    plt.show()

def parity(df, y_true_name, y_pred_name):
    y_true = df.loc[df["name"] == y_true_name, "y_predict"].tolist()[0][1:-1].split()
    y_pred = df.loc[df["name"] == y_pred_name, "y_predict"].tolist()[0][1:-1].split()

    y_true = [float(x) for x in y_true]
    y_pred = [float(x) for x in y_pred]
    
    i = 0
    while i < len(y_true):
        if y_pred[i] < 0 or y_pred[i] > 1000:
            y_true.pop(i)
            y_pred.pop(i)
        else:
            i += 1

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    
    min = 100
    max = 650
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0, "#ffffff"), (0., "#eeeeff"), (1, "#0000ff")])
    df.plot.hexbin(x="y_true", y="y_pred", gridsize=40, extent=[min, max, min, max], cmap=cmap)
    plt.xlabel("True Temps")
    plt.ylabel("Predicted Temps")
    plt.xlabel("Actual")
    plt.plot([min, max], [min, max], color="red")
    plt.savefig(y_pred_name + "_hexbin.png")
    plt.show()

if __name__ == "__main__":
    data_path = os.path.join(PATHNAME, sys.argv[1])
    df = pd.read_csv(data_path)
    type = sys.argv[2]
    model_list = []
    model = sys.argv[3] if len(sys.argv) > 3 else None
    if model == "all" and type == "parity":
        for row in df["model"]:
            if row != "true":
                model_list.append(row)
    else:
        model_list.append(model)

    if type == "r2":
        r2(df, "true")
    elif type == "ame":
        ame(df, "true")
    elif type == "parity":
        for m in model_list:
            parity(df, "true", m)
    

