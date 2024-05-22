import numpy as np
from sklearn.metrics import f1_score

pred = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
target = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




# Compute true positives, false positives, and false negatives
TP = np.sum((pred == 1) & (target == 1))+ np.sum((pred == 0) & (target == 0))
FP = np.sum((pred == 1) & (target == 0)) + np.sum((pred == 0) & (target == 1))
FN = np.sum((pred == 0) & (target == 1)) + np.sum((pred == 0) & (target == 1))

# Calculate precision, recall, and F1 score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy= TP/len(target)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)