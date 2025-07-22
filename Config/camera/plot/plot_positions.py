import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load JSON file
with open('../B02.json', 'r') as f:
    data = json.load(f)

# Extract camera positions and names
positions = []
names = []

for cam in data.get('cameras', []):
    pos = cam.get('Position', None)
    name = cam.get('Name', '')
    if pos and len(pos) == 3:
        positions.append(pos)
        names.append(name)

# Unpack x, y, z
x, y, z = zip(*positions)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='blue', s=50)

# Add labels
for name, xi, yi, zi in zip(names, x, y, z):
    ax.text(xi, yi, zi, name, fontsize=8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Positions in 3D')
ax.grid(True)

plt.show()
