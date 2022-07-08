import os
import sys
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def main(feature):
    models = ["random_forest", "gradient_boosting", "bagging"]
    splits = [(10, 0.1), (15, 0.15), (20, 0.2), (25, 0.25), (30, 0.3)]
    runs = range(5)

    r2_list = []
    mae_list = []

    for name, split in splits:
        r2_list.append({"test_split": split})
        mae_list.append({"test_split": split})
        for model in models:
            r2_sum = 0
            mae_sum = 0
            for run in runs:
                df = pd.read_csv(f"/Users/jayden/Documents/Code/MOFS_MRSEC/temp_testing/predictions/{feature}_{name}_{run}.csv")
                r2_sum += r2_score(df["true"], df[model])
                mae_sum += mean_absolute_error(df["true"], df[model])
            r2_sum /= len(runs)
            mae_sum /= len(runs)
            r2_list[-1][model] = r2_sum
            mae_list[-1][model] = mae_sum

    r2_df = pd.DataFrame(r2_list)
    mae_df = pd.DataFrame(mae_list)

    r2_df.to_csv(f"/Users/jayden/Documents/Code/MOFS_MRSEC/temp_testing/r2_scores/{feature}.csv", index=False)
    mae_df.to_csv(f"/Users/jayden/Documents/Code/MOFS_MRSEC/temp_testing/mae_scores/{feature}.csv", index=False)

if __name__ == "__main__":
    feature = sys.argv[1]
    main(feature)
