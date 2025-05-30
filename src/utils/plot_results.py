import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

with open("../../results.json") as f:
    data = json.load(f)

methods = [d["method"] for d in data]
accuracies = [d["accuracy"] for d in data]
f1s = [d["f1"] for d in data]
params = [d["trainable_params"] for d in data]

output_dir = "../../output/plots"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = os.path.join(output_dir, f"results_plot_{timestamp}.png")


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(methods, accuracies)
plt.title("Accuracy by Method")
plt.ylabel("Accuracy")

plt.subplot(1,2,2)
plt.bar(methods, f1s)
plt.title("F1-score by Method")
plt.ylabel("F1-score")

plt.tight_layout()
plt.savefig(plot_filename)  # timestamp
print(f"Plot saved as '{plot_filename}'")