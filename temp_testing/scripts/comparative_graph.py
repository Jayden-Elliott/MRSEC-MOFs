from cProfile import label
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATHNAME = os.path.abspath(".") + "/"

model_names = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "bagging": "Bagging"
}


def get_data(folder: str, dict: dict[str, dict[str, list[float]]], labels: list[str]):
    # feature_list = sorted(os.listdir(folder))
    # feature_list = ["rac.csv", "geometric.csv", "node_elements.csv", "linker.csv", "mofid_pubchem.csv", "homology.csv"]
    feature_list = ["rac.csv", "node_elements.csv", "rac_and_node_elements.csv"]
    for feature_set in feature_list:
        df = pd.read_csv(os.path.join(folder, feature_set))
        labels.append(feature_set.split(".")[0])

        for model_index in range(len(df.columns)):
            if df.columns[model_index] != "test_split":
                if df.columns[model_index] not in dict.keys():
                    dict[df.columns[model_index]] = {}

                for split_index in range(len(df.index)):
                    if df.iloc[split_index, 0] not in dict[df.columns[model_index]].keys():
                        dict[df.columns[model_index]][df.iloc[split_index, 0]] = []

                    dict[df.columns[model_index]][df.iloc[split_index, 0]].append(df.iloc[split_index, model_index])


def make_plots(output_folder: str, stat_short: str, stat: str, dict: dict[str, dict[str, list[float]]], labels: list[str]):
    for model, model_dict in dict.items():
        x = np.arange(len(labels))
        width = 0.75 / len(model_dict)
        fig, ax = plt.subplots()
        for i, (test_split, test_split_list) in enumerate(model_dict.items()):
            ax.bar(x + (i - 1) * width - 0.146, test_split_list, width, label=test_split)
        
        means = []
        for i in range(len(list(model_dict.values())[0])):
            means.append(np.mean([test_split_list[i] for test_split_list in model_dict.values()]))

        ax.bar(x, means, width=0.75, label="Mean", color="None", edgecolor="black")
        
        ax.set_ylabel(stat)
        ax.set_xlabel("Feature Set")
        ax.set_xticks(x, labels)
        ax.set_title(f"{model_names[model]} {stat}")
        ax.legend(loc="lower right", title="Test Split")
        
        fig.set_size_inches(2*len(labels), 6, forward=True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"{model}_{stat_short}.png"))
        plt.close(fig)

if __name__ == "__main__":
    output_folder = os.path.join(os.path.dirname(__file__), sys.argv[1])

    r2_folder = os.path.join(os.path.dirname(__file__), "r2_scores/")
    r2_labels = []
    r2_by_model = {}
    get_data(r2_folder, r2_by_model, r2_labels)
    make_plots(output_folder, "r2", "r² Value", r2_by_model, r2_labels)

    mae_folder = os.path.join(os.path.dirname(__file__), "mae_scores/")
    mae_labels = []
    mae_by_model = {}
    get_data(mae_folder, mae_by_model, mae_labels)
    make_plots(output_folder, "mae", "MAE", mae_by_model, mae_labels)