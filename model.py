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


def count_inliers(points, cx, cy, r, k=2.5, max_iter=3):
    residuals = [abs(math.hypot(x - cx, y - cy) - r) for (x, y) in points]
    for _ in range(max_iter):
        median = np.median(residuals)
        mad = np.median([abs(res - median) for res in residuals])
        sigma_hat = 1.4826 * mad
        threshold = k * sigma_hat
        inliers = [(x, y) for (x, y) in points if abs(math.hypot(x - cx, y - cy) - r) < threshold]
        residuals = [abs(math.hypot(x - cx, y - cy) - r) for (x, y) in inliers]
    return inliers, sigma_hat


def refine_circle_gauss_newton(points, cx, cy, r, max_iter=4):
    for it in range(max_iter):
        residuals = []
        J = []

        for (x, y) in points:
            d = math.hypot(x - cx, y - cy)
            # Residual
            f = d - r
            residuals.append(f)
            # Jacobian
            if d == 0:
                continue
            J.append([(cx - x) / d, (cy - y) / d, -1])

        residuals = np.array(residuals)
        J = np.array(J)

        # Update
        delta = np.linalg.lstsq(J, residuals, rcond=None)[0]
        cx -= delta[0]
        cy -= delta[1]
        r -= delta[2]
    return cx, cy, r


df = pd.read_csv("points.csv")
points = df[["x", "y"]].to_numpy()

n_iter = 5500
best_model = None
best_inliers = []
best_score = -float("inf")
number_of_inliers = 0
for _ in range(n_iter):
    (p1, p2, p3), result = choose_and_fit()
    if result is None:
        continue
    cx, cy, r = result
    inliers, sigma_hat = count_inliers(points, cx, cy, r)

    errors = [abs(math.hypot(x - cx, y - cy) - r) for (x, y) in inliers]
    mean_error = np.mean(errors) if errors else float("inf")
    score = len(inliers) - 1 * mean_error

    if best_model is None or score > best_score:
        best_score = score
        best_inliers = inliers
        best_model = cx, cy, r

print("Best circle:", best_model)
print("Number of inlier:", len(best_inliers), "/", len(points))
refined_cx, refined_cy, refined_r = refine_circle_gauss_newton(best_inliers, *best_model)
print("Refined circle:", refined_cx, refined_cy, refined_r)

fig, ax = plt.subplots()
for (x, y) in points:
    if (x, y) in best_inliers:
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
