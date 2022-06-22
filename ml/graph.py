import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error

PATHNAME = os.path.abspath(".") + "/"

csv_path = sys.argv[1]

def r2(df, y_true_name):
    scores = {}

    y_true = df.loc[df["model"] == y_true_name, "data"].tolist()[0][1:-1].split()
    for col in df.columns:
        if col != y_true_name:
            y_pred = df.loc[df["model"] == col, "data"].tolist()[0][1:-1].split()
            scores[col] = r2_score(y_true, y_pred)

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

    y_true = df.loc[df["model"] == y_true_name, "data"].tolist()[0][1:-1].split()
    for col in df.columns:
        if col != y_true_name:
            y_pred = df.loc[df["model"] == col, "data"].tolist()[0][1:-1].split()
            scores[col] = mean_absolute_error(y_true, y_pred)

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
    y_true = df.loc[df["model"] == y_true_name, "data"].tolist()[0][1:-1].split()
    y_pred = df.loc[df["model"] == y_pred_name, "data"].tolist()[0][1:-1].split()

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
    
    df.plot.hexbin(x="y_true", y="y_pred", gridsize=30, cmap="binary")
    plt.xlabel("True Temps")
    plt.ylabel("Predicted Temps")
    plt.xlabel("Actual")
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.savefig(y_pred_name + ".png")
    plt.show()

if __name__ == "__main__":
    data_path = os.path.join(PATHNAME, sys.argv[1])
    df = pd.read_csv(data_path)
    type = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else None
    if type == "r2":
        r2(df, "correct")
    elif type == "ame":
        ame(df, "correct")
    elif type == "parity":
        parity(df, "correct", model)
    

