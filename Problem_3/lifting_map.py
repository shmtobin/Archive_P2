# Problem 3: Lifting map and Delaunay triangulation

# a) Use the point cloud I gave in the section, plot the convex hull and generate
# the Delaunay triangulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# read the file
with open("mesh.dat", "r") as file:
    data = file.readlines()

# scatter plot and monotone chain from section
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
    plt.savefig('Plots/Hull_Delaunay_a1', bbox_inches='tight')

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

# compute Delaunay triangulation for coordinates
tri = Delaunay(coords)

# compute convex hull w/ monotone algorithm
hull = monotone_chain(coords)

# generate plot
scatter_plot(coords, convex_hull=hull, tri=tri)

# b) Now, lift the 2D mesh to the third dimension
# w/ z=x**2+y**2, calculate change in area and 
# plot it as a heat map.
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

# in debugging this heat map, the original values would lead to just a 
# pure black hull colored in, so to address this I capped the ratios 
# and used a log scale, because this gave the most desirable results

# cap maximum ratio to 100 (adjust based on your data distribution)
area_ratios_capped = np.clip(area_ratios, 0, 100)

# add small epsilon to avoid log(0)
area_ratios_log = np.log10(area_ratios_capped + 1e-6)
area_ratios_normalized = (area_ratios_log - np.min(area_ratios_log)) / (np.max(area_ratios_log) - np.min(area_ratios_log))
print("Area Ratios Normalized (3D / 2D):", area_ratios_normalized)

# plot the heatmap using tripcolor
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

plt.colorbar(trip, label='Area Ratio (3D / 2D), capped log scale')
plt.title('Change in Triangle Area After Parabolic Lifting')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig('Plots/area_change_b1', bbox_inches='tight')

# c) Calculate the induced metric, resulted by the lifting.

# induced metric = g [E, F, // F, G]
# E = σ_x*σ_x, F= σ_x*σ_y, G = σ_y*σ_y
# where σ_x and σ_y are the tangent vectors to the surface
# at a point (x, y)
# σ_x = ∂σ/∂x, σ_y = ∂σ/∂y
# σ(x,y) = [x // y // z]
# where z = x**2+y**2 as defined in this equation, giving 
# g=[1+4x^2, 4xy // 4xy, 1+4x^2]

def induced_metric(x, y):
    E = 1 + 4 * x**2
    F = 4 * x * y
    G = 1 + 4 * y**2
    return np.array([[E, F], [F, G]])

x = coords[:, 0]
y = coords[:, 1]

# calculating the induced metric for each point
g = np.array([induced_metric(xi, yi) for xi, yi in zip(x, y)])

# this would produce the induced metrix
# print("Induced metric:", g)
def metric_determinant(x, y):
    E = 1 + 4 * x**2
    F = 4 * x * y
    G = 1 + 4 * y**2
    return E * G - F**2

# compute determinant for all points
det_g = metric_determinant(coords[:, 0], coords[:, 1])
print("det_g:",det_g)

# plot as a heatmap
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=det_g, cmap='viridis')
plt.colorbar(label='Determinant of Metric Tensor (Area Distortion)')
plt.title('Determinant of Induced Metric Tensor')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Plots/metric_determinant_points_c1', bbox_inches='tight')
plt.show()


# d) calculate the surface normal of the lifted mesh and plot it on top of the mesh

def compute_surface_normals(coords_3d):
    """
    Compute unit surface normals for the parabolic surface z = x² + y².
    """
    x = coords_3d[:, 0]
    y = coords_3d[:, 1]
    
    # compute unnormalized normals: n = [-2x, -2y, 1]
    normals = np.zeros_like(coords_3d)
    normals[:, 0] = -2 * x
    normals[:, 1] = -2 * y
    normals[:, 2] = 1
    
    # normalize to unit length
    norms = np.linalg.norm(normals, axis=1)
    return normals / norms[:, np.newaxis]

def plot_mesh_with_normals(coords_3d, normals, tri):
    """Plot 3D mesh with surface normals."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot the surface w/ Delaunay triangulation
    ax.plot_trisurf(
        coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
        triangles=tri.simplices, cmap='viridis', alpha=0.7
    )
    
    # plot surface normals as arrows
    ax.quiver(
        coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        length=0.3, color='red', normalize=True
    )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lifted Mesh with Surface Normals')
    plt.show()
    plt.savefig('Plots/surface_normals_d1', bbox_inches='tight')

# compute surface normals
normals = compute_surface_normals(coords_3d)

# generate the plot
plot_mesh_with_normals(coords_3d, normals, tri)

# e) calculate the vertex normal of the lifted mesh and plot it on top of the mesh

def compute_vertex_normals(coords_3d, tri):
    """
    Compute vertex normals by averaging adjacent face normals.
    """
    num_vertices = coords_3d.shape[0]
    vertex_normals = np.zeros((num_vertices, 3))
    
    # calculate face normals for all triangles
    for simplex in tri.simplices:
        a, b, c = coords_3d[simplex]
        ab = b - a
        ac = c - a
        face_normal = np.cross(ab, ac)
        
        # normalize face normal
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal /= norm
            
        # accumulate to vertex normals
        for idx in simplex:
            vertex_normals[idx] += face_normal
    
    # normalize vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1)
    norms[norms == 0] = 1  # prevent division by zero
    return vertex_normals / norms[:, np.newaxis]

# compute vertex normals
vertex_normals = compute_vertex_normals(coords_3d, tri)

# plot mesh with vertex normals
plot_mesh_with_normals(coords_3d, vertex_normals, tri) 

# f) Compute the second fundamental form using the vertex normal.

# used FUNDAMENTAL FORMS OF SURFACES AND THE GAUSS-BONNET THEOREM by HUNTER S. CHASE
# notation


# analytical second derivatives of the parametric surface r(u,v) = (u, v, u**2 + v**2)
# The surface is defined by the parametrization:
#   r(u, v) = [u, v, u**2 + v**2]
#
# first derivatives:
#   r_u = ∂r/∂u = [1, 0, 2u]
#   r_v = ∂r/∂v = [0, 1, 2v]
#
# second derivatives:
#   r_uu = ∂**2r/∂u**2 = [0, 0, 2]  (since ∂/∂u of [1, 0, 2u] is [0, 0, 2])
#   r_uv = ∂**2r/∂u∂v = [0, 0, 0] (since ∂/∂v of [1, 0, 2u] is [0, 0, 0])
#   r_vv = ∂**2r/∂v**2 = [0, 0, 2]  (since ∂/∂v of [0, 1, 2v] is [0, 0, 2])
#
# these are constant for the parabolic surface z = x² + y².

r_uu = np.array([0.0, 0.0, 2.0])  # ∂**2r/∂u**2
r_uv = np.array([0.0, 0.0, 0.0])  # ∂**2r/∂u∂v
r_vv = np.array([0.0, 0.0, 2.0])  # ∂**2r/∂v**2

# compute second fundamental form coefficients for each vertex
# The second fundamental form is given by:
#   L = n * r_uu
#   M = n * r_uv
#   N = n * r_vv
# where n is the vertex normal.

L = np.array([np.dot(n, r_uu) for n in vertex_normals])
M = np.array([np.dot(n, r_uv) for n in vertex_normals])
N = np.array([np.dot(n, r_vv) for n in vertex_normals])

# visualization of L coefficient
# plot the L coefficient of the second fundamental form as a heatmap.
# this shows how the normal curvature varies across the surface.

plt.figure(figsize=(10, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=L, cmap='viridis', 
                s=50, edgecolor='none')
plt.colorbar(sc, label='Second Fundamental Form Coefficient L')
plt.title('Second Fundamental Form Coefficient L\n(Computed with Vertex Normals)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
plt.savefig('Plots/second_fundamental_form_f1', bbox_inches='tight')

#--------------------BEGIN WORK ON PART H---------------------------

# hb) Now, lift the 2D mesh to the third dimension
# w/ z=x**2+xy+y**2, calculate change in area and 
# plot it as a heat map.
def z(x, y):
    return x**2 +x*y+y**2

z_flat = z(coords[:, 0], coords[:, 1])
print("z_flat part h:", z_flat)
coords_3d = np.column_stack((coords, z_flat))
print("coords_3d part h:", coords_3d)

area_ratios = []
for simplex in tri.simplices:
    tri_2d = coords[simplex]
    tri_3d = coords_3d[simplex]

    area_2d = compute_triangle_area_2d(tri_2d)
    area_3d = compute_triangle_area_3d(tri_3d)
    
    area_ratios.append(area_3d / area_2d)

area_ratios = np.array(area_ratios)

print("Area Ratios (3D / 2D) part h:")
print(area_ratios)

# in debugging this heat map, the original values would lead to just a 
# pure black hull colored in, so to address this I capped the ratios 
# and used a log scale, because this gave the most desirable results

# cap maximum ratio to 100 (adjust based on your data distribution)
area_ratios_capped = np.clip(area_ratios, 0, 100)

# add small epsilon to avoid log(0)
area_ratios_log = np.log10(area_ratios_capped + 1e-6)
area_ratios_normalized = (area_ratios_log - np.min(area_ratios_log)) / (np.max(area_ratios_log) - np.min(area_ratios_log))
print("Area Ratios Normalized (3D / 2D) part h:", area_ratios_normalized)

# plot the heatmap using tripcolor
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

plt.colorbar(trip, label='Area Ratio (3D / 2D), capped log scale')
plt.title('Change in Triangle Area After Parabolic Lifting part h')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig('Plots/area_change_hb1', bbox_inches='tight')

# hc) Calculate the induced metric, resulted by the lifting.

# induced metric = g [E, F, // F, G]
# E = σ_x*σ_x, F= σ_x*σ_y, G = σ_y*σ_y
# where σ_x and σ_y are the tangent vectors to the surface
# at a point (x, y)
# σ_x = ∂σ/∂x, σ_y = ∂σ/∂y
# σ(x,y) = [x // y // z]
# where z = x**2+x*y+y**2 as defined in this equation, giving 
# g=[1+4x^2, 4xy // 4xy, 1+4x^2]

# hc) Calculate the induced metric
def induced_metric(x, y):
    # Partial derivatives of z = x² + xy + y²
    z_x = 2*x + y
    z_y = x + 2*y
    
    E = 1 + z_x**2
    F = z_x * z_y
    G = 1 + z_y**2
    return np.array([[E, F], [F, G]])

def metric_determinant(x, y):
    g = induced_metric(x, y)
    return g[0][0]*g[1][1] - g[0][1]**2

x = coords[:, 0]
y = coords[:, 1]

# calculating the induced metric for each point
g = np.array([induced_metric(xi, yi) for xi, yi in zip(x, y)])

# this would produce the induced metrix
# print("Induced metric:", g)
def metric_determinant(x, y):
    E = 1 + 4 * x**2
    F = 4 * x * y
    G = 1 + 4 * y**2
    return E * G - F**2

# compute determinant for all points
det_g = metric_determinant(coords[:, 0], coords[:, 1])
print("det_g part h:",det_g)

# plot as a heatmap
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], c=det_g, cmap='viridis')
plt.colorbar(label='Determinant of Metric Tensor (Area Distortion)')
plt.title('Determinant of Induced Metric Tensor part h')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Plots/metric_determinant_points_hc1', bbox_inches='tight')
plt.show()


# hd) calculate the surface normal of the lifted mesh and plot it on top of the mesh

def compute_surface_normals(coords_3d):
    x = coords_3d[:, 0]
    y = coords_3d[:, 1]
    
    # normal vector components for z = x² + xy + y²
    normals = np.zeros_like(coords_3d)
    normals[:, 0] = -(2*x + y)  # -∂z/∂x
    normals[:, 1] = -(x + 2*y)  # -∂z/∂y
    normals[:, 2] = 1
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1)
    return normals / norms[:, np.newaxis]

def plot_mesh_with_normals(coords_3d, normals, tri):
    """Plot 3D mesh with surface normals."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot the surface w/ Delaunay triangulation
    ax.plot_trisurf(
        coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
        triangles=tri.simplices, cmap='viridis', alpha=0.7
    )
    
    # plot surface normals as arrows
    ax.quiver(
        coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        length=0.3, color='red', normalize=True
    )
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lifted Mesh with Surface Normals part h')
    plt.show()
    plt.savefig('Plots/surface_normals_hd1', bbox_inches='tight')

# compute surface normals
normals = compute_surface_normals(coords_3d)

# generate the plot
plot_mesh_with_normals(coords_3d, normals, tri)

# he) calculate the vertex normal of the lifted mesh and plot it on top of the mesh

def compute_vertex_normals(coords_3d, tri):
    """
    Compute vertex normals by averaging adjacent face normals.
    """
    num_vertices = coords_3d.shape[0]
    vertex_normals = np.zeros((num_vertices, 3))
    
    # calculate face normals for all triangles
    for simplex in tri.simplices:
        a, b, c = coords_3d[simplex]
        ab = b - a
        ac = c - a
        face_normal = np.cross(ab, ac)
        
        # normalize face normal
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal /= norm
            
        # accumulate to vertex normals
        for idx in simplex:
            vertex_normals[idx] += face_normal
    
    # normalize vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1)
    norms[norms == 0] = 1  # prevent division by zero
    return vertex_normals / norms[:, np.newaxis]

# compute vertex normals
vertex_normals = compute_vertex_normals(coords_3d, tri)

# plot mesh with vertex normals
plot_mesh_with_normals(coords_3d, vertex_normals, tri)

# hf) Compute the second fundamental form using the vertex normal.

# used FUNDAMENTAL FORMS OF SURFACES AND THE GAUSS-BONNET THEOREM by HUNTER S. CHASE
# notation

# analytical second derivatives of the parametric surface r(u,v) = (u, v, u**2 + v**2)
# The surface is defined by the parametrization:
#   r(u, v) = [u, v, u**2 + v**2]
#
# first derivatives:
#   r_u = ∂r/∂u = [1, 0, 2u]
#   r_v = ∂r/∂v = [0, 1, 2v]
#
# second derivatives:
#   r_uu = ∂**2r/∂u**2 = [0, 0, 2]  (since ∂/∂u of [1, 0, 2u] is [0, 0, 2])
#   r_uv = ∂**2r/∂u∂v = [0, 0, 1] (since ∂/∂v of [1, 0, 2u] is [0, 0, 1])
#   r_vv = ∂**2r/∂v**2 = [0, 0, 2]  (since ∂/∂v of [0, 1, 2v] is [0, 0, 2])
#
# these are constant for the parabolic surface z = x² + x*y + y².

r_uu = np.array([0.0, 0.0, 2.0])   # ∂**2r/∂u**2
r_uv = np.array([0.0, 0.0, 1.0])  # ∂**2r/∂u∂v
r_vv = np.array([0.0, 0.0, 2.0])  # ∂**2r/∂v**2

# compute second fundamental form coefficients for each vertex
# The second fundamental form is given by:
#   L = n * r_uu
#   M = n * r_uv
#   N = n * r_vv
# where n is the vertex normal.

L = np.array([np.dot(n, r_uu) for n in vertex_normals])
M = np.array([np.dot(n, r_uv) for n in vertex_normals])
N = np.array([np.dot(n, r_vv) for n in vertex_normals])

# visualization of L coefficient
# plot the L coefficient of the second fundamental form as a heatmap.
# this shows how the normal curvature varies across the surface.

plt.figure(figsize=(10, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=L, cmap='viridis',
                s=50, edgecolor='none')
plt.colorbar(sc, label='Second Fundamental Form Coefficient L')
plt.title('Second Fundamental Form L part h')
plt.savefig('Plots/second_fundamental_form_hf1', bbox_inches='tight')