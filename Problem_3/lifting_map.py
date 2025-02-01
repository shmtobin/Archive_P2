# Problem 3: Lifting map and Delaunay triangulation

# a) Use the point cloud I gave in the section, plot the convex hull and generate
# the Delaunay triangulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Read the file
with open("mesh.dat", "r") as file:
    data = file.readlines()

def scatter_plot(coords, convex_hull=None, tri=None):
    xs, ys = zip(*coords)
    plt.scatter(xs, ys, color='k')  # Plot points in black

    # plot convex hull
    if convex_hull is not None:
        for i in range(len(convex_hull)):
            c0 = convex_hull[i]
            c1 = convex_hull[(i+1) % len(convex_hull)]
            plt.plot([c0[0], c1[0]], [c0[1], c1[1]], 'r-', lw=2, label='Convex Hull' if i == 0 else "")

    # plot Delaunay triangulation
    if tri is not None:
        plt.triplot(coords[:,0], coords[:,1], tri.simplices, color='b', lw=0.5, linestyle='-', marker='', label='Delaunay Triangulation')

    plt.legend()
    plt.show()

def monotone_chain(points):
    points = sorted(map(tuple, points), key=lambda p: (p[0], p[1]))
    lower = []
    for p in points:
        while len(lower) >= 2 and (lower[-1][0] - lower[-2][0])*(p[1] - lower[-2][1]) - (lower[-1][1] - lower[-2][1])*(p[0] - lower[-2][0]) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and (upper[-1][0] - upper[-2][0])*(p[1] - upper[-2][1]) - (upper[-1][1] - upper[-2][1])*(p[0] - upper[-2][0]) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])

# extract coordinates
x, y = [], []
for line in data:
    if line.strip() == "" or line.startswith('X'):
        continue
    columns = line.split()
    try:
        x.append(float(columns[0]))
        y.append(float(columns[1]))
    except ValueError:
        continue
coords = np.array(list(zip(x, y)))

# compute Delaunay triangulation
tri = Delaunay(coords)

# compute convex hull
hull = monotone_chain(coords)

# generate plot
# scatter_plot(coords, convex_hull=hull, tri=tri)

def z(x, y):
    return x**2 + y**2

z_flat = z(coords[:, 0], coords[:, 1])
print("z_flat:", z_flat)
coords_3d = np.column_stack((coords, z_flat))
print("coords_3d:", coords_3d)

def compute_triangle_area_2d(points):
    """compute area of a 2D triangle using the cross product."""
    a, b, c = points
    return 0.5 * np.abs(np.cross(b - a, c - a))

def compute_triangle_area_3d(points):
    """compute area of a 3D triangle using the cross product."""
    a, b, c = points
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross)

area_ratios = []
for simplex in tri.simplices:
    tri_2d = coords[simplex]
    tri_3d = coords_3d[simplex]

    area_2d = compute_triangle_area_2d(tri_2d)
    area_3d = compute_triangle_area_3d(tri_3d)
    
    area_ratios.append(area_3d / area_2d)

area_ratios = np.array(area_ratios)

print("Area Ratios (3D / 2D):")
print(area_ratios)

# in debugging this, the original values would lead to just a pure black 
# hull colored in, so to address this I capped the ratios and used a 
# log scale

# add small epsilon to avoid log(0)
# cap maximum ratio to 100 (adjust based on your data distribution)
area_ratios_capped = np.clip(area_ratios, 0, 100)
# add small epsilon to avoid log(0)
area_ratios_log = np.log10(area_ratios_capped + 1e-6)
area_ratios_normalized = (area_ratios_log - np.min(area_ratios_log)) / (np.max(area_ratios_log) - np.min(area_ratios_log))
print("Area Ratios Normalized (3D / 2D):")
print(area_ratios_normalized)

# Plot the heatmap using tripcolor
plt.figure(figsize=(10, 8))
trip = plt.tripcolor(
    coords[:, 0], 
    coords[:, 1], 
    tri.simplices, 
    facecolors=area_ratios_normalized, 
    edgecolors='none',  
    cmap='inferno',     
    shading='flat'    
)

plt.colorbar(trip, label='Area Ratio (3D / 2D)')
plt.title('Change in Triangle Area After Parabolic Lifting')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# c) Calculate the induced metric, resulted by the lifting.

# induced metric = g [E, F, // F, G]
# E = σ_x*σ_x, F= σ_x*σ_y, G = σ_y*σ_y
# where σ_x and σ_y are the tangent vectors to the surface
# at a point (x, y)
# σ_x = ∂σ/∂x, σ_y = ∂σ/∂y
# σ(x,y) = [x // y // z]
# where z = x**2+y**2 as defined in this equation, giving 
# g=[1+4x^2, 4xy // 4xy, 1+4x^2]

def metric_determinant(x, y):
    E = 1 + 4 * x**2
    F = 4 * x * y
    G = 1 + 4 * y**2
    return E * G - F**2

# Compute determinant for all points
det_g = metric_determinant(coords[:, 0], coords[:, 1])

# Plot as a heatmap
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=det_g, cmap='viridis')
plt.colorbar(label='Determinant of Metric Tensor (Area Distortion)')
plt.title('Determinant of Induced Metric Tensor')
plt.xlabel('x')
plt.ylabel('y')
plt.show()