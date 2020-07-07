import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# SET-MLP
for i in range(5):
    with open('../Pretrained_results/set_mlp_5000_training_samples_e13_rand' + str(i) + '_monitor.json') as json_file:
        data = np.array(json.load(json_file))

        if i == 0:
            memory_usage = np.zeros((1650, 1))
            cpu_usage = np.zeros((1650, 1))
            number_of_threads = np.zeros((1650, 1))

        for idx, p in enumerate(data):
            memory_usage[idx] += p['memory_usage']
            cpu_usage[idx] += p['cpu_usage']
            number_of_threads[idx] += p['n_threads']

memory_usage_set_mlp = pd.Series(memory_usage.reshape(-1) / 5)
cpu_usage_set_mlp = pd.Series(cpu_usage.reshape(-1) / 5)
number_of_threads_set_mlp = pd.Series(number_of_threads.reshape(-1) / 5)


# Dense-MLP
for i in range(5):
    with open('../Pretrained_results/dense_mlp_5000_training_samples_rand' + str(i) + '_monitor.json') as json_file:
        data = np.array(json.load(json_file))
        if i == 0:
            memory_usage = np.zeros((8782, 1))
            cpu_usage = np.zeros((8782, 1))
            number_of_threads = np.zeros((8782, 1))

        for idx, p in enumerate(data):
            # print(idx)
            memory_usage[idx] += p['memory_usage']
            cpu_usage[idx] += p['cpu_usage']
            number_of_threads[idx] += p['n_threads']

memory_usage_dense_mlp = pd.Series(memory_usage.reshape(-1) / 5)
cpu_usage_dense_mlp = pd.Series(cpu_usage.reshape(-1) / 5)
number_of_threads_dense_mlp = pd.Series(number_of_threads.reshape(-1) / 5)


# FixProb-MLP
for i in range(5):
    with open('../Pretrained_results/fixprob_mlp_5000_training_samples_e13_rand' + str(i) + '_monitor.json') as json_file:
        data = np.array(json.load(json_file))

        if i == 0:
            memory_usage = np.zeros((1616, 1))
            cpu_usage = np.zeros((1616, 1))
            number_of_threads = np.zeros((1616, 1))

        for idx, p in enumerate(data):
            # print(idx)
            memory_usage[idx] += p['memory_usage']
            cpu_usage[idx] += p['cpu_usage']
            number_of_threads[idx] += p['n_threads']

memory_usage_fixprob_mlp = pd.Series(memory_usage.reshape(-1) / 5)
cpu_usage_fixprob_mlp = pd.Series(cpu_usage.reshape(-1) / 5)
number_of_threads_fixprob_mlp = pd.Series(number_of_threads.reshape(-1) / 5)


font = {'size': 9}
fig = plt.figure(figsize=(10, 5))
matplotlib.rc('font', **font)
fig.subplots_adjust(wspace=0.2, hspace=0.05)

ax1 = fig.add_subplot(1,2,1)
ax1.plot(np.arange(0, len(memory_usage_set_mlp)/2, 0.5), memory_usage_set_mlp.rolling(window=5).mean(), label="SET-MLP memory usage", color="r")
ax1.plot(np.arange(0, len(memory_usage_fixprob_mlp)/2, 0.5), memory_usage_fixprob_mlp.rolling(window=5).mean(), label="MLP$_{FixProb}$ memory usage", color="g")
ax1.plot(np.arange(0, len(memory_usage_dense_mlp)/2, 0.5), memory_usage_dense_mlp.rolling(window=5).mean(), label="FC-MLP memory usage", color="y")
ax1.grid(True)
ax1.set_ylabel("Fashion MNIST\nMemory usage [%]")
ax1.set_xlabel("Time [s]")
ax1.legend(loc=4, fontsize=8)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.arange(0, len(cpu_usage_set_mlp)/2, 0.5), cpu_usage_set_mlp.rolling(window=5).mean(), label="SET-MLP cpu usage", color="r")
ax2.plot(np.arange(0, len(cpu_usage_fixprob_mlp)/2, 0.5), cpu_usage_fixprob_mlp.rolling(window=5).mean(), label="MLP$_{FixProb}$ cpu usage", color="g")
ax2.plot(np.arange(0, len(cpu_usage_dense_mlp)/2, 0.5), cpu_usage_dense_mlp.rolling(window=5).mean(), label="FC-MLP cpu usage", color="y")
ax2.grid(True)
ax2.set_ylabel("Fashion MNIST\nCPU usage [%]")
ax2.set_xlabel("Time [s]")
ax2.legend(loc=1,fontsize=8)

samples = 5000
plt.savefig("../Results/fashionmnist_memory_and_cpu_usage_samples"+str(samples)+".pdf", bbox_inches='tight')

plt.close()
