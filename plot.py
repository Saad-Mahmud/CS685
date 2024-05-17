import numpy as np
import matplotlib.pyplot as plt

# Data
models = ["Mamba1.4B", "Phi1.5B", "Gemma2B", "RGemma2B", "Mamba2.8B", 
          "Zephyr3B", "Mamba7B", "Llama7B", "Llama8B", "Mistral7B", "Gemma7B"]
sensitivity = np.array([
    [0.00498904, 0.00321812, 0.01014795, 0.00607136],
    [0.0213155,  0.01252051, 0.02123292, 0.0054206 ],
    [0.00567389, 0.00777554, 0.0046499, 0.00802673],
    [0.01484942, 0.0074414,  0.0124021,  0.01043014],
    [0.00881973, 0.00266066, 0.008413,   0.00405323],
    [0.07467147, 0.01298639, 0.00595173, 0.00936997],
    [0.00855012, 0.00288885, 0.0027417,  0.00476298],
    [0.01601523, 0.0091432,  0.03052186, 0.01390872],
    [0.02268373, 0.02401513, 0.01575327, 0.01879708],
    [0.01542464, 0.00592397, 0.01633978, 0.01315273],
    [0.03442361, 0.04616662, 0.03540142, 0.04451822],
])

attributes = ['Sex', 'Age', 'Race', 'Sexuality']
bar_width = 0.2

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(sensitivity.shape[1]):
    plt.bar(np.arange(len(models)) + i * bar_width, sensitivity[:, i], width=bar_width, label=attributes[i])

plt.xlabel('Models')
plt.ylabel('Shapley Sensitivity')
plt.title('Sensitivity of LLMs to Different Attributes')
plt.xticks(np.arange(len(models)) + bar_width * 1.5, models, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('fairnes.pdf')
plt.show()
