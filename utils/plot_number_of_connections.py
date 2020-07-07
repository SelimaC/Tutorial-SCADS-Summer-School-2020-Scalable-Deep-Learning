import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 9}
fig = plt.figure(figsize=(10, 5))
matplotlib.rc('font', **font)
fig.subplots_adjust(wspace=0.2, hspace=0.05)

set_mlp_parameters = 8.5249
fixprob_mlp_parameters = 8.5249
dense_mlp_parameters = 279.4000

objects = ('SET-MLP', 'MLP$_{FixProb}$', 'Dense-MLP')
y_pos = np.arange(len(objects))
plt.bar(0, set_mlp_parameters, label="SET-MLP", color="r")
plt.bar(1, fixprob_mlp_parameters, label="MLP$_{FixProb}$", color="g")
plt.bar(2, dense_mlp_parameters, label="Dense-MLP", color="y")
plt.grid(True)
plt.ylabel("Number of weights (x10$^{5}$)")
plt.xticks(y_pos, objects)
plt.legend(loc=4, fontsize=8)

samples = 5000
plt.savefig("../Results/fashionmnist_compare_number_of_parameters.pdf", bbox_inches='tight')

plt.close()
