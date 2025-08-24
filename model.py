import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def fit_circle_3pts(p1, p2, p3):
    # Cholinergic control
    A = np.array([[p1[0], p1[1], 1],
                  [p2[0], p2[1], 1],
                  [p3[0], p3[1], 1]])
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    e = ((x1 ** 2 - x2 ** 2) + (y1 ** 2 - y2 ** 2)) / 2.0
    f = ((x1 ** 2 - x3 ** 2) + (y1 ** 2 - y3 ** 2)) / 2.0
    det = a * d - b * c
    cx = (d * e - b * f) / det
    cy = (-c * e + a * f) / det
    r = math.hypot(x1 - cx, y1 - cy)
    return cx, cy, r


def choose_and_fit():
    df = pd.read_csv("points.csv")
    sample = df.sample(n=3)[["x", "y"]].to_numpy()
    p1, p2, p3 = sample[0], sample[1], sample[2]
    result = fit_circle_3pts(p1, p2, p3)
    if result is None:
        return (p1, p2, p3), None
    else:
        return (p1, p2, p3), result

def count_inliers(points,cx,cy,r, threshold=5):
    inliers = []
    for (x,y) in points:
        d=math.hypot(x-cx,y-cy)
        error=abs(d-r)
        if error < threshold:
            inliers.append((x,y))
    return inliers

def refine_circle_least_squares(points):
    X = []
    Y = []
    for (x,y) in points:
        X.append([x, y, 1])
        Y.append(-(x**2 + y**2))
    M = np.array(X)
    b = np.array(Y)

    params, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    A, B, C = params

    cx = -A / 2
    cy = -B / 2
    r = math.sqrt(cx**2 + cy**2 - C)
    return cx, cy, r

df=pd.read_csv("points.csv")
points=df[["x","y"]].to_numpy()

n_iter=5500
best_model = None
best_inliers = []
number_of_inliers = 0
for _ in range(n_iter):
    (p1, p2, p3), result = choose_and_fit()
    if result is None:
        continue
    cx, cy, r = result
    inliers = count_inliers(points, cx, cy, r)
    if len(inliers) > number_of_inliers:
        number_of_inliers = len(inliers)
        best_inliers = inliers
        best_model = cx, cy, r

print("Best circle:", best_model)
print("Number of inlier:", len(best_inliers), "/", len(points))
refined_cx, refined_cy, refined_r = refine_circle_least_squares(best_inliers)
print("Refined circle:", refined_cx, refined_cy, refined_r)

fig, ax = plt.subplots()
for (x,y) in points:
    if (x,y) in best_inliers:
        ax.plot(x, y, "bo", markersize=4)  # inlier
    else:
        ax.plot(x, y, "ro", markersize=4)  # outlier
refined_patch = Circle((refined_cx, refined_cy), refined_r,
                       fill=False, color="green", linewidth=2, linestyle="--", label="Refined circle")
ax.add_patch(refined_patch)
ax.plot(refined_cx, refined_cy, "black", markersize=8, label="Refined center")
"""ax.plot(p1[0], p1[1],"ro", markersize=4)
ax.plot(p2[0], p2[1],"ro", markersize=4)
ax.plot(p3[0], p3[1],"ro", markersize=4)

circle_patch = Circle((cx, cy), r, fill=False, color="black", linewidth=2)
ax.add_patch(circle_patch)
ax.plot(cx, cy, "kx", markersize=6)"""

ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect("equal", adjustable="box")
plt.show()