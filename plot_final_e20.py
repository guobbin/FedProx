import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils.utils import smooth
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)


def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []

    for line in open(file_name, 'r'):

        search_train_accu = re.search(r'At round (.*) training accuracy: (.*)', line, re.M | re.I)
        if search_train_accu:
            rounds.append(int(search_train_accu.group(1)))
        else:
            search_test_accu = re.search(r'At round (.*) accuracy: (.*)', line, re.M | re.I)
            if search_test_accu:
                accu.append(float(search_test_accu.group(2)))

        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M | re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: (.*)', line, re.M | re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))

    return rounds, sim, loss, accu


f = plt.figure(figsize=[13, 20])

# log = ["synthetic_1_1", "mnist", "femnist", "shakespeare", "sent140_772user"]
log = ["cnn2m"]
# titles = ["Synthetic", "MNIST", "FEMNIST", "Shakespeare", "Sent140"]
titles = ["MNIST"]
# rounds = [200, 100, 200, 40, 800]
rounds = [200, 100, 200, 40, 800]
# mus=[1, 1, 1, 0.001, 0.01]
mus = [1, 1, 1, 0.001, 0.01]
drop_rates = [0, 0.5, 0.9]

sampling_rate = [1, 1, 2, 1, 10]
smooth_weight = 0.90

labels = [r'FedProx ($\mu$=0)', r'FedProx ($\mu$=0.1)', r'FedProx ($\mu$=00.1)', r'FedProx ($\mu$=0.10)', r'FedProx ($\mu$=00.10.10)']

improv = 0


for drop_rate in range(1):
    for idx in range(1):
        for ind in range(3):
            ax = plt.subplot(3, 1, ind+1)
            rounds1, sim1, losses1, test_accuracies1 = parse_log(log[idx] + "/fedprox200_drop"+str(drop_rates[drop_rate])+"_mu0")
            rounds2, sim2, losses2, test_accuracies2 = parse_log(log[idx] + "/fedprox200_drop"+str(drop_rates[drop_rate])+"_mu0.1")
            rounds3, sim3, losses3, test_accuracies3 = parse_log(log[idx] + "/fedprox200_drop"+str(drop_rates[drop_rate])+"_mu00.1")
            rounds4, sim4, losses4, test_accuracies4 = parse_log(log[idx] + "/fedprox200_drop"+str(drop_rates[drop_rate])+"_mu0.10")
            rounds5, sim5, losses5, test_accuracies5 = parse_log(log[idx] + "/fedprox200_drop"+str(drop_rates[drop_rate])+"_mu00.10.10")



            # if sys.argv[1] == 'loss':
            if ind == 0:
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]),
                         smooth(np.asarray(losses1)[::sampling_rate[idx]], smooth_weight), ":", linewidth=1.0, color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]),
                         smooth(np.asarray(losses2)[::sampling_rate[idx]], smooth_weight), '--', linewidth=1.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]),
                         smooth(np.asarray(losses3)[::sampling_rate[idx]], smooth_weight), linewidth=1.0, color="#17becf")
                plt.plot(np.asarray(rounds4[:len(losses4):sampling_rate[idx]]),
                         smooth(np.asarray(losses4)[::sampling_rate[idx]], smooth_weight), ':',  linewidth=1.0, color="#a7becf")
                plt.plot(np.asarray(rounds5[:len(losses5):sampling_rate[idx]]),
                         smooth(np.asarray(losses5)[::sampling_rate[idx]], smooth_weight), '-.',  linewidth=1.0, color="#57becf")

            elif ind == 2:
                plt.plot(np.asarray(rounds1[:len(sim1):sampling_rate[idx]]),
                         smooth(np.asarray(sim1)[::sampling_rate[idx]], smooth_weight), ":", linewidth=1.0, color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(sim2):sampling_rate[idx]]),
                         smooth(np.asarray(sim2)[::sampling_rate[idx]], smooth_weight), '--', linewidth=1.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(sim3):sampling_rate[idx]]),
                         smooth(np.asarray(sim3)[::sampling_rate[idx]], smooth_weight), linewidth=1.0, color="#17becf")
                plt.plot(np.asarray(rounds4[:len(sim4):sampling_rate[idx]]),
                         smooth(np.asarray(sim4)[::sampling_rate[idx]], smooth_weight), ':', linewidth=1.0, color="#a7becf")
                plt.plot(np.asarray(rounds5[:len(sim5):sampling_rate[idx]]),
                         smooth(np.asarray(sim5)[::sampling_rate[idx]], smooth_weight), '-.', linewidth=1.0, color="#57becf")

            elif ind == 1:
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]),
                         smooth(np.asarray(test_accuracies1)[::sampling_rate[idx]], smooth_weight), ":", linewidth=1.0, color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]),
                         smooth(np.asarray(test_accuracies2)[::sampling_rate[idx]], smooth_weight), '--', linewidth=1.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]),
                         smooth(np.asarray(test_accuracies3)[::sampling_rate[idx]], smooth_weight), linewidth=1.0, color="#17becf")
                plt.plot(np.asarray(rounds4[:len(test_accuracies4):sampling_rate[idx]]),
                         smooth(np.asarray(test_accuracies4)[::sampling_rate[idx]], smooth_weight), ':', linewidth=1.0, color="#a7becf")
                plt.plot(np.asarray(rounds5[:len(test_accuracies5):sampling_rate[idx]]),
                         smooth(np.asarray(test_accuracies5)[::sampling_rate[idx]], smooth_weight), '-.', linewidth=1.0, color="#57becf")

            # plt.xlabel("# Rounds", fontsize=22)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)

            if ind == 0 and idx == 0:
                plt.ylabel('Training Loss', fontsize=10)
            elif ind == 1 and idx == 0:
                plt.ylabel('Testing Accuracy', fontsize=10)
            elif ind == 2 and idx == 0:
                plt.ylabel('Dissimilar', fontsize=10)
            if ind == 0:
                plt.title(titles[idx], fontsize=10, fontweight='bold')

            ax.tick_params(color='#dddddd')
            ax.spines['bottom'].set_color('#dddddd')
            ax.spines['top'].set_color('#dddddd')
            ax.spines['right'].set_color('#dddddd')
            ax.spines['left'].set_color('#dddddd')
            ax.set_xlim(0, rounds[idx])


f.legend(frameon=False, loc='lower center', ncol=5, prop=dict(weight='bold'), borderaxespad=-0.3, fontsize=20, labels=labels)  # note: different from plt.legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
f.savefig("loss_accuracy_full_"+log[0]+'200.pdf')
