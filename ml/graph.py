import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

data = pd.read_csv(sys.argv[1])
data = data.values
actual = data[:, 0].tolist()
predicted = data[:, 1].tolist()

print("R2: {}".format(r2_score(actual, predicted)))
print("MAE: {}".format(mean_absolute_error(actual, predicted)))

plt.scatter(actual, predicted)
plt.plot([0, 800], [0, 800], color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
