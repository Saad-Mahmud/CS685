import matplotlib.pyplot as plt
import numpy as np

models = ["Mamba2.8B", "Zephyr3B", "Llama8B", "Mamba7B"]

# Performance metrics for with CoT
with_cot = {
    "train": [0.5, 0.39, 0.53, 0.7],
    "test": [0.17, 0.16, 0.29, 0.12],
    "ood": [0.07, 0.02, 0.075, 0.02]
}

# Performance metrics for without CoT
without_cot = {
    "train": [0.39, 0.3, 0.42, 0.47],
    "test": [0.09, 0.13, 0.21, 0.1],
    "ood": [0.09, 0.06, 0.08, 0.04]
}

x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

# Plotting bars for each dataset and accuracy type
rects1 = ax.bar(x - width, with_cot["train"], width, label='With CoT - Train')
rects2 = ax.bar(x, with_cot["test"], width, label='With CoT - Test')
rects3 = ax.bar(x + width, with_cot["ood"], width, label='With CoT - OOD')

rects4 = ax.bar(x - width, without_cot["train"], width,  label='Without CoT - Train', alpha=0.5)
rects5 = ax.bar(x, without_cot["test"], width,  label='Without CoT - Test', alpha=0.5)
rects6 = ax.bar(x + width, without_cot["ood"], width,  label='Without CoT - OOD', alpha=0.5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models',fontsize=16)
ax.set_ylabel('Accuracy',fontsize=16)
ax.set_title('Comparison of Train, Test, and OOD Accuracy with and without CoT',fontsize=16)
ax.set_xticks(x,)
ax.set_xticklabels(models, rotation=45, ha="right",fontsize=16)
ax.legend(fontsize=16)

fig.tight_layout()

plt.savefig('CoT.pdf')
#plt.show()