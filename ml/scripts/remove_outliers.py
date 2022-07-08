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

csv = sys.argv[1]
df = pd.read_csv(os.path.join(PATHNAME, csv))
correct = df.data.tolist()[0][1:-1].split()
predicted = df.data.tolist()[int(sys.argv[2])][1:-1].split()

correct = [float(x) for x in correct]
predicted = [float(x) for x in predicted]

i = 0
while i < len(correct):
    if predicted[i] < 0 or predicted[i] > 100000:
        correct.pop(i)
        predicted.pop(i)
    else:
        i += 1

print(r2_score(correct, predicted))
print(mean_absolute_error(correct, predicted))

plt.scatter(correct, predicted)
plt.plot([0, 10000], [0, 10000], color="red")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim(0, 10000)
plt.ylim(0, 10000)
plt.savefig(os.path.join(PATHNAME, f"{sys.argv[2]}.png"))
plt.show()