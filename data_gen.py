import random as rn
import matplotlib.pyplot as plt

class Circle:
    def __init__(self):
        self.center_x = rn.uniform(-50, 50)
        self.center_y = rn.uniform(-50, 50)
        self.radius = rn.uniform(50, 100)

circles = [Circle() for _ in range(3)]

fig, ax = plt.subplots()

for c in circles:
    circle_patch = plt.Circle((c.center_x, c.center_y), c.radius,fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(circle_patch)
    ax.plot(c.center_x,c.center_y,'ro')

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()