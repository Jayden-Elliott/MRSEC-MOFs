import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file1 = os.path.join(os.path.abspath("."), sys.argv[1])
df1 = pd.read_csv(file1)
col_name1 = sys.argv[2]
col_list1 = list(df1[col_name1])

file2 = os.path.join(os.path.abspath("."), sys.argv[3])
df2 = pd.read_csv(file2)
col_name2 = sys.argv[4]
col_list2 = list(df2[col_name2])

n1, bins1, patches1 = plt.hist(col_list1, bins=80, range=(0, 800))
n2, bins2, patches2 = plt.hist(col_list2, bins=80, range=(0, 800))
plt.clf()

diff = [a - b for a, b in zip(n1, n2)]
print(diff)
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.ylim(top=80, bottom=-80)
plt.title(f"{col_name1} - {col_name2}")
plt.bar(np.linspace(0, 800, 80, endpoint=False), diff, width=10)

plt.savefig(f"{sys.argv[5]}.png")
plt.show()
