import numpy as np                     
import os                             
import time                           
import matplotlib.pyplot as plt       


from sklearn.datasets import fetch_covtype                  
from sklearn.model_selection import train_test_split, learning_curve  
from sklearn.preprocessing import StandardScaler            
from sklearn.tree import DecisionTreeClassifier               
from sklearn.neighbors import KNeighborsClassifier            
from sklearn.svm import SVC                                  
from sklearn.neural_network import MLPClassifier             

# Metrics for evaluation
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


# SETUP

# Ensures "results" folder exists for saving plots
os.makedirs("results", exist_ok=True)


# 1. LOAD DATA

print("Loading dataset...")

# Load dataset (downloaded from sklearn)
# X = features (54 columns)
# y = labels (7 classes)
X, y = fetch_covtype(return_X_y=True)

print("Dataset shape:", X.shape)

# 2. SAMPLE (STRATIFIED)

# Reduce dataset to 20,000 rows (assignment requirement)
# stratify=y ensures class proportions are preserved
X_sample, _, y_sample, _ = train_test_split(
    X, y,
    train_size=20000,
    stratify=y,
    random_state=42
)

print("Sample shape:", X_sample.shape)

# 3. TRAIN TEST SPLIT

# Split into training (80%) and test (20%)
# stratify again → keeps class distribution consistent
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample,
    test_size=0.2,
    stratify=y_sample,
    random_state=42
)

# 4. SCALING

# StandardScaler → mean=0, std=1
# Must only fit on training data to avoid leakage
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn scaling
X_test_scaled = scaler.transform(X_test)         # apply scaling

# EVALUATION FUNCTION

def evaluate(model, X, y):
    # Predict labels
    preds = model.predict(X)

    # Return key evaluation metrics
    return (
        accuracy_score(y, preds),       # overall accuracy
        f1_score(y, preds, average='macro'),  # equal weight per class
        balanced_accuracy_score(y, preds)     # handles imbalance
    )


# 5. DECISION TREE TUNING + CURVE

# Testing different depths (controls model complexity)
depths = [5, 10, 20]
dt_scores = []

print("\nDT Complexity Curve...")

# Manual hyperparameter loop
for d in depths:

    dt = DecisionTreeClassifier(max_depth=d, random_state=42)

    dt.fit(X_train, y_train)

    # Evaluate performance
    acc, f1, bal = evaluate(dt, X_test, y_test)

    dt_scores.append(f1)

    print(f"Depth {d} -> F1: {f1:.3f}")


# Plot Decision Tree complexity curve
plt.plot(depths, dt_scores, marker='o')
plt.xlabel("Max Depth")
plt.ylabel("Macro F1")
plt.title("Decision Tree Complexity Curve")

# Save plot to file (used in report)
plt.savefig("results/dt_curve.png")

# Close plot (does not display on screen)
plt.close()


# Train final DT using chosen depth
dt_best = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_best.fit(X_train, y_train)

# 6. KNN TUNING + CURVE

# k controls neighborhood size
# small k → low bias, high variance
# large k → high bias, smoother boundaries
k_values = [3, 5, 15]
knn_scores = []

print("\nkNN Complexity Curve...")

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    # IMPORTANT: must use scaled data (distance-based)
    knn.fit(X_train_scaled, y_train)

    acc, f1, bal = evaluate(knn, X_test_scaled, y_test)

    knn_scores.append(f1)

    print(f"k={k} -> F1: {f1:.3f}")


# Plot kNN curve
plt.plot(k_values, knn_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Macro F1")
plt.title("kNN Complexity Curve")

plt.savefig("results/knn_curve.png")
plt.close()


# Final chosen kNN model
knn_best = KNeighborsClassifier(n_neighbors=5)
knn_best.fit(X_train_scaled, y_train)

# 7. SVM TUNING + CURVE

# C controls regularization strength
C_values = [0.1, 1, 10]
svm_scores = []

print("\nSVM Complexity Curve...")

for c in C_values:

    svm = SVC(kernel='rbf', C=c, gamma='scale')

    # MUST use scaled data (SVM sensitive to scale)
    svm.fit(X_train_scaled, y_train)

    acc, f1, bal = evaluate(svm, X_test_scaled, y_test)

    svm_scores.append(f1)

    print(f"C={c} -> F1: {f1:.3f}")


# Plot SVM curve
plt.plot(C_values, svm_scores, marker='o')
plt.xlabel("C")
plt.ylabel("Macro F1")
plt.title("SVM Complexity Curve")

plt.savefig("results/svm_curve.png")
plt.close()


# Final SVM model (NOTE: you found C=10 was best, but using C=1 here)
svm_best = SVC(kernel='rbf', C=1, gamma='scale')
svm_best.fit(X_train_scaled, y_train)

# 8. NEURAL NETWORK

print("\nTraining Neural Network...")

# MLP neural network
# Constraint: SGD only (no momentum)
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),   # one hidden layer
    solver='sgd',               # required optimizer
    momentum=0,                 # must be 0
    learning_rate_init=0.01,
    max_iter=200,
    random_state=42
)

