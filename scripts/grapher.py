import csv
import numpy as np

input_file = "ramnsrall.csv"

generations_rew = {}
generations_nov = {}

with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        generation = int(row['Generation'])
        reward = float(row['Reward'])
        novelty = float(row['Novelty'])

        if generation not in generations_rew:
            generations_rew[generation] = []

        if generation not in generations_nov:
            generations_nov[generation] = []

        generations_rew[generation].append(reward)
        generations_nov[generation].append(novelty)

for generation in generations_rew.keys():
    generations_rew[generation] = np.array(generations_rew[generation])

for generation in generations_nov.keys():
    generations_nov[generation] = np.array(generations_nov[generation])

average_rewards = np.zeros((len(generations_rew),))
max_rewards = np.zeros((len(generations_rew),))
average_novelties = np.zeros((len(generations_rew),))
max_novelties = np.zeros((len(generations_rew),))

for generation in generations_rew.keys():
    average_rewards[generation] = generations_rew[generation].mean()
    max_rewards[generation] = generations_rew[generation].max()
    average_novelties[generation] = generations_nov[generation].mean()
    max_novelties[generation] = generations_nov[generation].max()

import matplotlib.pyplot as plt

plt.plot(range(0, len(generations_nov)), average_rewards, label='average reward')
plt.plot(range(0, len(generations_nov)), max_rewards, label='max reward')
#plt.plot(range(0, len(generations_nov)), average_novelties, label='average novelty')
#plt.plot(range(0, len(generations_nov)), max_novelties, label='max novelty')

plt.legend()
plt.xlabel('generations')

plt.savefig("figure.png")

plt.show()


