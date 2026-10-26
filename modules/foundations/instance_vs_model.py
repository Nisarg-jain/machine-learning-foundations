import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 1. Feature matrix: [CGPA, IQ]
X_train = np.array([
    [8.5, 115],  # Student 1: Placed
    [8.0, 108],  # Student 2: Placed
    [7.8, 110],  # Student 3: Placed
    [5.5, 90],   # Student 4: Not Placed
    [6.0, 95],   # Student 5: Not Placed
    [5.2, 88],   # Student 6: Not Placed
])

# Labels: 1 = Placed, 0 = Not Placed
y_train = np.array([1, 1, 1, 0, 0, 0])

# New test candidate to evaluate
query_student = np.array([[7.5, 103]])

# ==========================================
# 1. INSTANCE-BASED LEARNING (KNN)
# ==========================================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)  # Stores raw observations in memory

knn_pred = knn.predict(query_student)
distances, indices = knn.kneighbors(query_student)

print("--- INSTANCE-BASED LEARNING (KNN) ---")
print(f"Prediction for [CGPA=7.5, IQ=103]: {'PLACED' if knn_pred[0] == 1 else 'NOT PLACED'}")
print(f"Indices of 3 Nearest Neighbors: {indices[0]}")
print(f"Distances to Neighbors: {np.round(distances[0], 2)}")

# ==========================================
# 2. MODEL-BASED LEARNING (LOGISTIC REGRESSION)
# ==========================================
lr = LogisticRegression()
lr.fit(X_train, y_train)  # Learns optimal decision boundary weights

lr_pred = lr.predict(query_student)
w1, w2 = lr.coef_[0]
b = lr.intercept_[0]

print("\n--- MODEL-BASED LEARNING (LOGISTIC REGRESSION) ---")
print(f"Prediction for [CGPA=7.5, IQ=103]: {'PLACED' if lr_pred[0] == 1 else 'NOT PLACED'}")
print(f"Learned Decision Boundary: {w1:.3f}*(CGPA) + {w2:.3f}*(IQ) + ({b:.3f}) = 0")