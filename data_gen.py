import random as rn
import matplotlib.pyplot as plt
import math
from statistics import mean, stdev

class Circle:
    def __init__(self):
        self.center_x = rn.uniform(-50, 50)
        self.center_y = rn.uniform(-50, 50)
        self.radius = rn.uniform(50, 100)

    def generate_points_on_circle(self,a,b,R,n_points):
        points=[]
        for _ in range(n_points):
            theta=rn.uniform(0,2*math.pi)
            x=a+R*math.cos(theta)
            y=b+R*math.sin(theta)
            points.append((x,y))
        return points
    
    def add_noise_to_points(self, points, sigma):
        noisy_points = []
        for (x, y) in points:
            noise_x = x + rn.gauss(0, sigma)
            noise_y = y + rn.gauss(0, sigma)
            noisy_points.append((noise_x, noise_y))
        return noisy_points
    
    def noise_stats(self,a,b,R,points):
        residuals = []
        for x, y in points:
            d=math.hypot(x - a, y - b)
            residuals.append(d-R)
        abs_res=[abs(r) for r in residuals]
        stats={
            "n": len(points),
            "signed_mean": mean(residuals),
            "signed_std": stdev(residuals) if len(points) > 1 else 0.0,
            "abs_mean": mean(abs_res),
            "abs_std": stdev(abs_res) if len(points) > 1 else 0.0,
            "min_signed": min(residuals),
            "max_signed": max(residuals),
            # yarıçapa göre oransal (yüzde)
            "abs_std_pct_of_R": (mean(abs_res)/R)*100 if R != 0 else float("nan"),
            "p95_abs": sorted(abs_res)[int(0.95*(len(abs_res)-1))] if points else 0.0
        }
        return stats
    
    def get_sigma_from_radius(self):
        return self.radius * 0.04
    
def generate_outliers(n_points,x_range=(-200, 200), y_range=(-200, 200)):
    outliers = []
    for _ in range(n_points):
        x = rn.uniform(*x_range)
        y = rn.uniform(*y_range)
        outliers.append((x, y))
    return outliers

colours=['blue','red','orange','black']
circles = [Circle() for _ in range(1)]
fig, ax = plt.subplots()
legend_labels=[]

for i, c in enumerate(circles, start=1):
    circle_colour=rn.choice(colours)
    circle_patch = plt.Circle((c.center_x, c.center_y), c.radius,fill=False, edgecolor=circle_colour, linewidth=1)

    #ax.add_patch(circle_patch)
    #ax.plot(c.center_x,c.center_y,'bo',markersize=4)
    #ax.text(c.center_x+3,c.center_y+3,  f"({c.center_x:.1f}, {c.center_y:.1f})", fontsize=8, color='black')

    legend_labels.append((f"Circle {i} - R={c.radius:.1f}", circle_colour))

    noisy_points = c.add_noise_to_points(c.generate_points_on_circle(c.center_x, c.center_y, c.radius, 20), c.get_sigma_from_radius())
    for x, y in noisy_points:
            ax.plot(x,y,'ro',markersize=4)
    stats=c.noise_stats(c.center_x, c.center_y, c.radius, noisy_points)
    print(stats)

for label, color in legend_labels:
    ax.plot([], [], color=color, label=label)

outliers= generate_outliers(80)
for x, y in outliers:
    ax.plot(x, y, 'ro', markersize=4)

ax.axhline(0, color='black', linewidth=1) 
ax.axvline(0, color='black', linewidth=1)

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()