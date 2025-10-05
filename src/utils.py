"""utils.py
Helper functions for plotting and evaluation.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score




def plot_elbow(inertias, ks, out_path=None):
plt.figure()
plt.plot(ks, inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method for K selection")
if out_path:
plt.savefig(out_path)
else:
plt.show()




def compute_silhouette(X, labels):
if len(set(labels)) == 1:
return float('nan')
return silhouette_score(X, labels)
