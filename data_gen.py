import random as rn
import matplotlib.pyplot as plt
import math
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
    
colours=['blue','red','orange','black']
circles = [Circle() for _ in range(3)]
fig, ax = plt.subplots()
legend_labels=[]

for i, c in enumerate(circles, start=1):
    circle_colour=rn.choice(colours)
    circle_patch = plt.Circle((c.center_x, c.center_y), c.radius,fill=False, edgecolor=circle_colour, linewidth=1)

    ax.add_patch(circle_patch)
    ax.plot(c.center_x,c.center_y,'ro',markersize=4)
    ax.text(c.center_x+3,c.center_y+3,  f"({c.center_x:.1f}, {c.center_y:.1f})", fontsize=8, color='black')

    legend_labels.append((f"Circle {i} - R={c.radius:.1f}", circle_colour))

    for x, y in c.generate_points_on_circle(c.center_x,c.center_y,c.radius,10):
            ax.plot(x,y,'ro',markersize=4)

for label, color in legend_labels:
    ax.plot([], [], color=color, label=label)

ax.axhline(0, color='black', linewidth=1) 
ax.axvline(0, color='black', linewidth=1)

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal', adjustable='box')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()