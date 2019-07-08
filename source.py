#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4321753/

import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew

import matplotlib.pyplot as plt
plt.style.use('ggplot')

data = np.random.normal(0, 1, 10000)


plt.hist(data, bins=60)
plt.show()
print("mean: ", np.mean(data))
print("var: ", np.var(data))
print("skew: ", skew(data))
print("kurt: ", kurtosis(data))


data = np.append(data, [1000])

plt.hist(data, bins=60)
plt.show()

print("mean: ", np.mean(data))
print("var: ", np.var(data))
print("skew: ", skew(data))
print("kurt: ", kurtosis(data))
