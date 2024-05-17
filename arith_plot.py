import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["Mamba1.4B", "Phi1.5B", "Gemma2B", "RGemma2B", "Mamba2.8B", 
          "Zephyr3B", "Mamba7B", "Llama7B", "Llama8B", "Mistral7B", "Gemma7B"]
train_acc = [0.26, 0.13, 0.21, 0.13, 0.33, 0.28, 0.53, 0.19, 0.42, 0.26, 0.28]
test_acc = [0.23, 0.11, 0.19, 0.15, 0.24, 0.22, 0.28, 0.19, 0.27, 0.24, 0.22]
ood_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
zeroshot_acc = [0, 0, 0, 0.01, 0.01, 0, 0, 0, 0.02, 0, 0.06]

x = np.arange(len(models))

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(x - bar_width, train_acc, bar_width, alpha=opacity, color='b', label='Train Accuracy')
rects2 = plt.bar(x, test_acc, bar_width, alpha=opacity, color='g', label='Test Accuracy')
rects3 = plt.bar(x + bar_width, ood_acc, bar_width, alpha=opacity, color='r', label='OOD Accuracy')
rects4 = plt.bar(x + 2 * bar_width, zeroshot_acc, bar_width, alpha=opacity, color='y', label='Zeroshot Accuracy')


plt.xlabel('Models', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(x - 2*bar_width, models,fontsize=16, rotation=45)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.title('Model Accuracy Comparison Multiplication_Small Dataset', fontsize=16)


plt.tight_layout()
plt.savefig("mul_small.pdf")
#plt.show()
plt.close()

train_acc = [0.07, 0.03, 0.07, 0.03, 0.09, 0.08, 0.12, 0.06, 0.12, 0.08, 0.08]
test_acc =  [0.04, 0.02, 0.03, 0.03, 0.05, 0.03, 0.07, 0.04, 0.07, 0.05, 0.05]
ood_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
zeroshot_acc = [0, 0, 0, 0.00, 0.00, 0, 0, 0, 0.00, 0, 0.02]

x = np.arange(len(models))

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(x - bar_width, train_acc, bar_width, alpha=opacity, color='b', label='Train Accuracy')
rects2 = plt.bar(x, test_acc, bar_width, alpha=opacity, color='g', label='Test Accuracy')
rects3 = plt.bar(x + bar_width, ood_acc, bar_width, alpha=opacity, color='r', label='OOD Accuracy')
rects4 = plt.bar(x + 2 * bar_width, zeroshot_acc, bar_width, alpha=opacity, color='y', label='Zeroshot Accuracy')


plt.xlabel('Models', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(x -2*bar_width, models,fontsize=16, rotation=45)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.title('Model Accuracy Comparison Multiplication_Large Dataset', fontsize=16)


plt.tight_layout()
plt.savefig("mul_large.pdf")
#plt.show()
plt.close()

train_acc = [0.06, 0.02, 0.99, 0.82, 0.12, 0.21, 0.23, 1, 1, 1, 1]
test_acc = [0.01, 0, 0.95, 0.74, 0.02, 0.07, 0.02, 0.97, 0.92, 0.97, 1]
ood_acc = [0, 0, 0.82, 0.18, 0, 0, 0, 0.75, 0.64, 0.92, 1]
zeroshot_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.05, 0.08]

x = np.arange(len(models))

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(x - bar_width, train_acc, bar_width, alpha=opacity, color='b', label='Train Accuracy')
rects2 = plt.bar(x, test_acc, bar_width, alpha=opacity, color='g', label='Test Accuracy')
rects3 = plt.bar(x + bar_width, ood_acc, bar_width, alpha=opacity, color='r', label='OOD Accuracy')
rects4 = plt.bar(x + 2 * bar_width, zeroshot_acc, bar_width, alpha=opacity, color='y', label='Zeroshot Accuracy')


plt.xlabel('Models', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(x - 2*bar_width, models,fontsize=16, rotation=45)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.title('Model Accuracy Comparison Addition_Large Dataset', fontsize=16)


plt.tight_layout()
plt.savefig("add_large.pdf")
plt.close()


train_acc = [0.7, 0.22, 1, 0.98, 0.91, 0.99, 0.98, 1, 1, 1, 1]
test_acc = [0.55, 0.27, 1, 0.92, 0.7, 0.81, 0.65, 0.97, 1, 1, 1]
ood_acc = [0, 0, 0.35, 0.63, 0, 0, 0, 0.44, 0.82, 1, 0.98]
zeroshot_acc = [0, 0, 0, 0.06, 0.07, 0, 0, 0.11, 0.42, 0.4, 0.23]

x = np.arange(len(models))

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(x - bar_width, train_acc, bar_width, alpha=opacity, color='b', label='Train Accuracy')
rects2 = plt.bar(x, test_acc, bar_width, alpha=opacity, color='g', label='Test Accuracy')
rects3 = plt.bar(x + bar_width, ood_acc, bar_width, alpha=opacity, color='r', label='OOD Accuracy')
rects4 = plt.bar(x + 2 * bar_width, zeroshot_acc, bar_width, alpha=opacity, color='y', label='Zeroshot Accuracy')

plt.xlabel('Models', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(x - 2*bar_width, models,fontsize=16, rotation=45)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.title('Model Accuracy Comparison Addition Small Dataset', fontsize=16)
plt.tight_layout()
plt.savefig("add_small.pdf")
