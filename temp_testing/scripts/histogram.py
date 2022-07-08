import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = os.path.join(os.path.abspath("."), sys.argv[1])
df = pd.read_csv(file)
col_name = sys.argv[2]
col_list = list(df[col_name])

plt.hist(col_list, bins=80, range=(0, 800))
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.title(f"{col_name}")

plt.savefig(f"{sys.argv[3]}.png")
plt.show()
