import random as rn
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from statistics import mean, stdev


class Circle:
    def __init__(self):
        self.center_x = rn.uniform(-50, 50)
        self.center_y = rn.uniform(-50, 50)
        self.radius = rn.uniform(50, 100)

    def generate_points_on_circle(self, a, b, R, n_points):
        points = []
        for _ in range(n_points):
            theta = rn.uniform(0, 2 * math.pi)
            x = a + R * math.cos(theta)
            y = b + R * math.sin(theta)
            points.append((x, y))
        return points

    def noise_stats(self, a, b, R, points):
        residuals = []
        for x, y in points:
            d = math.hypot(x - a, y - b)
            residuals.append(d - R)
        abs_res = [abs(r) for r in residuals]
        stats = {
            "n": len(points),
            "signed_mean": mean(residuals),
            "signed_std": stdev(residuals) if len(points) > 1 else 0.0,
            "abs_mean": mean(abs_res),
            "abs_std": stdev(abs_res) if len(points) > 1 else 0.0,
            "min_signed": min(residuals),
            "max_signed": max(residuals),
            "abs_std_pct_of_R": (mean(abs_res) / R) * 100 if R != 0 else float("nan"),
            "p95_abs": (
                sorted(abs_res)[int(0.95 * (len(abs_res) - 1))] if points else 0.0
            ),
        }
        return stats

    def get_sigma_from_radius(self):
        return self.radius * 0.04


def create_csv_file():
    df = pd.DataFrame({"id": id_numbers, "x": x_list, "y": y_list})

    df.to_csv("points.csv", index=False, encoding="utf-8")
    print("CSV file created: points.csv")


def get_id_for_points(points):
    id = rn.randint(1000, 9999)
    while id in points:
        id = rn.randint(1000, 9999)
    points.append(id)
    return id


def add_noise_to_points(points, sigma):
    noisy_points = []
    for x, y in points:
        noise_x = x + rn.gauss(0, sigma)
        noise_y = y + rn.gauss(0, sigma)
        noisy_points.append((noise_x, noise_y))
    return noisy_points


def generate_outliers(circle_n, hits_n):
    real_hits = circle_n * hits_n
    change_ratio = rn.uniform(-0.3, 0.3)
    number_of_outliers = int((int(real_hits * 1.2)) * (1 + change_ratio))

    outliers_80 = []  # 80% of outliers
    for _ in range(int(number_of_outliers * 0.8)):
        x = rn.uniform(-200, 200)
        y = rn.uniform(-200, 200)
        outliers_80.append((x, y))

    outliers_20 = []  # 20% of outliers
    for _ in range(int(number_of_outliers * 0.2)):
        r_max = rn.uniform(20, 30)
        r = rn.uniform(0, r_max)
        theta = rn.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        outliers_20.append((x, y))

    outliers_line = []  # Line outliers
    number_of_line_outliers = int(number_of_outliers * 0.3)
    y_values = np.linspace(200, -200, number_of_line_outliers)
    x_values = np.full(number_of_line_outliers, -180)
    outliers_line = list(zip(x_values, y_values))
    outliers_line_noisy = add_noise_to_points(outliers_line, 3)
    # Combine all outliers
    outliers = outliers_80 + outliers_20 + outliers_line_noisy
    return outliers


id_numbers = []
x_list = []
y_list = []
number_of_circles = 3
number_of_hits_on_circle = 20
colours = ["blue", "red", "orange", "black"]
circles = [Circle() for _ in range(number_of_circles)]
fig, ax = plt.subplots()
legend_labels = []
for i, c in enumerate(circles, start=1):
    circle_colour = rn.choice(colours)
    circle_patch = plt.Circle(
        (c.center_x, c.center_y),
        c.radius,
        fill=False,
        edgecolor=circle_colour,
        linewidth=1,
    )

    # ax.add_patch(circle_patch)
    # ax.plot(c.center_x,c.center_y,'bo',markersize=4)
    # ax.text(c.center_x+3,c.center_y+3,  f"({c.center_x:.1f}, {c.center_y:.1f})", fontsize=8, color='black')

    legend_labels.append((f"Circle {i} - R={c.radius:.1f}", circle_colour))

    noisy_points = add_noise_to_points(
        c.generate_points_on_circle(
            c.center_x, c.center_y, c.radius, number_of_hits_on_circle
        ),
        c.get_sigma_from_radius(),
    )

    for x, y in noisy_points:
        ax.plot(x, y, "ro", markersize=4)
        id_number = get_id_for_points(id_numbers)
        x_list.append(x)
        y_list.append(y)
        # ax.text(x+3,y+3,  f"id = {id_number}", fontsize=8, color='black')
        # print(f"id = {id_number} x = {x:.2f} y = {y:.2f}")

    stats = c.noise_stats(c.center_x, c.center_y, c.radius, noisy_points)
    print(stats)

for label, color in legend_labels:
    ax.plot([], [], color=color, label=label)

outliers = generate_outliers(number_of_circles, number_of_hits_on_circle)
for x, y in outliers:
    ax.plot(x, y, "ro", markersize=4)
    id_number = get_id_for_points(id_numbers)
    x_list.append(x)
    y_list.append(y)
    # ax.text(x+3,y+3,  f"id = {id_number}", fontsize=8, color='black')
    # print(f"id = {id_number} x = {x:.2f} y = {y:.2f}")

create_csv_file()

ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect("equal", adjustable="box")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()
