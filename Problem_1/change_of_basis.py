# Problem 1: Coordinate transformation and Parallel Transport

# a)  Write down the coefficients of three coordinate systems 
# that describe a point (1, θ, ϕ) on the unit sphere.

# Cartesian -> Spherical: (x, y, z) -> (sqrt(x^2 + y^2 + z^2), cos^-1(z / sqrt(x^2 + y^2 + z^2)), tan^-1(y / x))
# Cartesian -> Cylindrical: (x, y, z) -> (sqrt(x^2 + y^2), tan^-1(y / x), z)

# Spherical -> Cartesian: (r, θ, ϕ) -> (r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ))
# Spherical -> Cylindrical: (r, θ, ϕ) -> (r * sin(θ), ϕ, r * cos(θ))

# Cylindrical -> Cartesian: (ρ, ϕ, z) -> (ρ * cos(ϕ), ρ * sin(ϕ), z)
# Cylindrical -> Spherical: (ρ, ϕ, z) -> (sqrt(ρ^2 + z^2), tan^-1(ρ / z), ϕ)


# Write a python function or multiple python functions that convert 
# coordinates and basis between the three.

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# if elif structure for all coordinate transformations in one function
def convert_coordinates(x1, x2, x3, coordinate_input, coordinate_output):
    if coordinate_input == "cartesian":
        x, y, z = x1, x2, x3
        
        if coordinate_output == "cylindrical":
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            return r, theta, z
        
        elif coordinate_output == "spherical":
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r) if r != 0 else 0
            return r, theta, phi
    
    elif coordinate_input == "cylindrical":
        r, theta, z = x1, x2, x3
        
        if coordinate_output == "cartesian":
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return x, y, z
        
        elif coordinate_output == "spherical":
            rho = np.sqrt(r**2 + z**2)
            phi = np.arctan2(r, z) if rho != 0 else 0
            return rho, theta, phi
    
    elif coordinate_input == "spherical":
        r, theta, phi = x1, x2, x3
        
        if coordinate_output == "cartesian":
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            return x, y, z
        
        elif coordinate_output == "cylindrical":
            rho = r * np.sin(phi)
            z = r * np.cos(phi)
            return rho, theta, z
    
    if coordinate_input == coordinate_output:
        print("nice conversion bucko")
        return x1, x2, x3  # explicitly return the input coordinates
    
    else:
        raise ValueError("Invalid coordinate system specified.")

# example conversions :D
cart_to_cart = convert_coordinates(3, 4, 5, "cartesian", "cartesian")
print(f"Input: (3, 4, 5) in Cartesian -> Output: {cart_to_cart} in Cartesian")
cart_to_sph = convert_coordinates(3, 4, 5, "cartesian", "spherical")
print(f"Input: (3, 4, 5) in Cartesian -> Output: {cart_to_sph} in Spherical")
cyl_to_cart = convert_coordinates(3, 4, 5, "cylindrical", "cartesian")
print(f"Input: (3, 4, 5) in Cylindrical -> Output: {cyl_to_cart} in Cartesian")
sph_to_cyl = convert_coordinates(3, 4, 5, "spherical", "cylindrical")
print(f"Input: (3, 4, 5) in Spherical -> Output: {sph_to_cyl} in Cylindrical")

# b) A position vector in the unit sphere is r = e_r such that the coordinate
# (1, θ, ϕ). Create local orthonormal coordinate systems on the unit sphere and
# represent them as vectors in Cartesian coordinate system.

def spherical_basis_vectors(theta, phi):
    # radial unit vector (e_r)
    e_r = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # polar unit vector (e_theta)
    e_theta = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    
    # azimuthal unit vector (e_phi)
    e_phi = np.array([
        -np.sin(phi),
        np.cos(phi),
        0
    ])
    
    return e_r, e_theta, e_phi

# plot the unit sphere with radial position vectors
def plot_unit_sphere_with_vectors():
    # create a grid of theta and phi values
    theta = np.linspace(0, np.pi, 20)  # polar angle (0 to pi)
    phi = np.linspace(0, 2 * np.pi, 20)  # azimuthal angle (0 to 2pi)
    
    # convert spherical coordinates to Cartesian coordinates
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    # create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot the unit sphere
    ax.plot_surface(x, y, z, color='b', alpha=0.3)
    
    # plot radial position vectors at selected points
    for i in range(0, len(theta), 5):
        for j in range(0, len(phi), 5):
            e_r, e_theta, e_phi = spherical_basis_vectors(theta[i], phi[j])
            x0, y0, z0 = np.sin(theta[i]) * np.cos(phi[j]), np.sin(theta[i]) * np.sin(phi[j]), np.cos(theta[i])
            
            # plot the radial vector
            ax.quiver(x0, y0, z0, e_r[0], e_r[1], e_r[2], color='r', length=0.1, normalize=True)
            
            # plot the polar vector
            ax.quiver(x0, y0, z0, e_theta[0], e_theta[1], e_theta[2], color='g', length=0.1, normalize=True)
            
            # plot the azimuthal vector
            ax.quiver(x0, y0, z0, e_phi[0], e_phi[1], e_phi[2], color='b', length=0.1, normalize=True)
    
    # set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Unit Sphere with Local Orthonormal Basis Vectors')
    
    plt.show()
    plt.savefig('Plots/unit_sphere_with_vectors_b1.png')

plot_unit_sphere_with_vectors()

# c) Can you plot the unit sphere in spherical basis {e_r,e_θ,e_ϕ} (1, θ, ϕ)? If
# so, plot it. If not, explain why. It should be very simple, and don’t over think.

# the unit sphere cannot be directly plotted in the spherical basis {e_r, e_θ, e_ϕ}
#  because these basis vectors are coordinate-dependent and change direction at 
# every point (θ, ϕ). Instead, we can represent the sphere in Cartesian coordinates 
# and overlay the spherical basis vectors at selected points to visualize their 
# orientation.

# d) Create a function that generates the local coordinate system on a given
# mesh, parametrized by a general surface z = f(x, y).

# generates the local coordinate system (tangent vectors and normal vector) 
# on a surface z = f(x, y).
def local_coordinate_system(f, x, y, dx=1e-5, dy=1e-5):
    # f = surface function
    # x, y = grid of points with step sizes 1e-5

    # compute the partial derivatives of f(x, y)
    df_dx, df_dy = np.gradient(f(x, y), dx, dy)
    
    # compute the tangent vectors
    T_x = np.stack([np.ones_like(x), np.zeros_like(x), df_dx], axis=-1)  # T_x = (1, 0, df/dx)
    T_y = np.stack([np.zeros_like(x), np.ones_like(x), df_dy], axis=-1)  # T_y = (0, 1, df/dy)
    
    # compute the normal vector as the cross product of T_x and T_y
    N = np.cross(T_x, T_y)
    
    # normalize the normal vector to get the unit normal vector e_r
    e_r = N / np.linalg.norm(N, axis=-1, keepdims=True)
    
    return T_x, T_y, N, e_r

# example usage
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))  # Example surface: z = sin(sqrt(x^2 + y^2))

# define a grid of x and y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# compute the local coordinate system
T_x, T_y, N, e_r = local_coordinate_system(f, X, Y)

# print results at a specific point (e.g., x=1, y=1)
idx = (np.abs(x - 1).argmin(), np.abs(y - 1).argmin())
# print statements that work but don't want to keep spamming
# print("Tangent vector T_x at (1, 1):", T_x[idx])
# print("Tangent vector T_y at (1, 1):", T_y[idx])
# print("Normal vector N at (1, 1):", N[idx])
# print("Unit normal vector e_r at (1, 1):", e_r[idx])

# e) Write a code that demonstrate the parallel transport of a vector
# n(θ_0, ϕ = 0) near the north pole [r = 1, ϕ = 0, θ = π/5] to the equator at
# [r = 1, ϕ = 0, θ = π/2] following the unit-speed parametrization


# initial conditions for vector perpendicular to motion
theta0 = np.pi / 5  # near north pole
alpha = 0.0          # component along e_θ (coordinate basis)
beta = 1.0 / np.sin(theta0)  # component along e_ϕ (normalized to unit length)

# generate points along the path γ(θ) = (θ, ϕ=0), θ from θ0 to π/2
theta_vals = np.linspace(theta0, np.pi/2, 20)
phi_vals = np.zeros_like(theta_vals)

# Cartesian coordinates of points on the path
x = np.sin(theta_vals) * np.cos(phi_vals)
y = np.sin(theta_vals) * np.sin(phi_vals)
z = np.cos(theta_vals)

# parallel transport: compute transported vector components (in spherical basis)
V_theta = alpha * np.ones_like(theta_vals)  # V_θ remains constant
V_phi = beta * np.sin(theta0) / np.sin(theta_vals)  # V_ϕ adjusts to maintain norm

# convert spherical basis vectors to Cartesian components
# corrected e_ϕ components with sin(theta)
Vx = V_theta * np.cos(theta_vals) * np.cos(phi_vals) + V_phi * (-np.sin(theta_vals) * np.sin(phi_vals))
Vy = V_theta * np.cos(theta_vals) * np.sin(phi_vals) + V_phi * (np.sin(theta_vals) * np.cos(phi_vals))
Vz = V_theta * (-np.sin(theta_vals)) + V_phi * 0.0

# plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# draw unit sphere
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)

# plot the transported vectors (scale for visibility)
scale = 0.3
ax.quiver(x, y, z, scale*Vx, scale*Vy, scale*Vz, color='red', 
          label='Parallel Transported Vector', normalize=False)

# highlight the path (meridian)
ax.plot(x, y, z, 'b-', linewidth=2, label='Transport Path (ϕ=0)')

ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
ax.set_title(f'Parallel Transport from θ={theta0:.2f} to Equator')
ax.legend()
plt.savefig('Plots/unit_speed_pt_e1', bbox_inches='tight')
plt.show()

# initial conditions for vector parallel with transport
theta0 = np.pi / 5  # near north pole
alpha = 1.0          # component along e_θ (coordinate basis)
beta = 0.0           # component along e_ϕ (normalized to unit length)

# generate fewer points along the path γ(θ) = (θ, ϕ=0), θ from θ0 to π/2
num_points = 10  # reduced number of vectors for clarity
theta_vals = np.linspace(theta0, np.pi/2, num_points)
phi_vals = np.zeros_like(theta_vals)

# cartesian coordinates of points on the path
x = np.sin(theta_vals) * np.cos(phi_vals)
y = np.sin(theta_vals) * np.sin(phi_vals)
z = np.cos(theta_vals)

# parallel transport: compute transported vector components (in spherical basis)
V_theta = alpha * np.ones_like(theta_vals)  # V_θ remains constant
V_phi = beta * np.ones_like(theta_vals)     # V_ϕ remains zero

# convert spherical basis vectors to Cartesian components
# corrected e_θ and e_ϕ components
Vx = V_theta * np.cos(theta_vals) * np.cos(phi_vals)  # e_θ_x
Vy = V_theta * np.cos(theta_vals) * np.sin(phi_vals)  # e_θ_y
Vz = V_theta * (-np.sin(theta_vals))                  # e_θ_z

# plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# draw unit sphere (optional, comment out if distracting)
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)

# plot the transported vectors (scale for visibility)
scale = 0.4  # decreased scale for clarity
ax.quiver(x, y, z, scale*Vx, scale*Vy, scale*Vz, color='red', 
          label='Parallel Transported Vector', normalize=False, arrow_length_ratio=0.1)

# highlight the path (meridian)
ax.plot(x, y, z, 'b-', linewidth=2, label='Transport Path (ϕ=0)')

ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
ax.set_title(f'Parallel Transport from θ={theta0:.2f} to Equator')
ax.legend()
ax.view_init(elev=20, azim=-45)  
plt.savefig('Plots/unit_speed_pt_e2', bbox_inches='tight')
plt.show()

# f) write a code that demonstrate the parallel transport of a 
# vector from [r = 1, ϕ = 0, θ = θ0] to [r = 1, ϕ = 2π, θ = θ0]

# parameters
theta0 = np.pi/4  # Constant polar angle (latitude)
alpha = 1.0        # Initial component along e_θ
beta = 0.0         # Initial component along e_ϕ
n_mag = 1.0        # Vector magnitude

# generate points along the path γ(ϕ) = (θ=θ0, ϕ), ϕ from 0 to 2π
phi_vals = np.linspace(0, 2*np.pi, 30)
theta_vals = theta0 * np.ones_like(phi_vals)

# cartesian coordinates of the path
x = np.sin(theta0) * np.cos(phi_vals)
y = np.sin(theta0) * np.sin(phi_vals)
z = np.cos(theta0) * np.ones_like(phi_vals)

# parallel transport: Vector rotates by holonomy angle Δ = 2π(1 - cosθ0)
delta = 2 * np.pi * (1 - np.cos(theta0))  # Total rotation angle

# transported vector components (in rotating spherical basis)
V_theta = alpha * np.cos(delta * phi_vals/(2*np.pi)) - beta * np.sin(delta * phi_vals/(2*np.pi))
V_phi = alpha * np.sin(delta * phi_vals/(2*np.pi)) + beta * np.cos(delta * phi_vals/(2*np.pi))

# spherical basis vectors in Cartesian coordinates
def spherical_to_cartesian(theta, phi):
    e_theta = np.array([
        np.cos(theta)*np.cos(phi),
        np.cos(theta)*np.sin(phi),
        -np.sin(theta)
    ])
    e_phi = np.array([
        -np.sin(phi),
        np.cos(phi),
        0
    ])
    return e_theta, e_phi

# compute cartesian components of transported vectors
Vx, Vy, Vz = [], [], []
for i in range(len(phi_vals)):
    e_theta, e_phi = spherical_to_cartesian(theta0, phi_vals[i])
    V_cart = V_theta[i]*e_theta + V_phi[i]*e_phi
    Vx.append(V_cart[0])
    Vy.append(V_cart[1])
    Vz.append(V_cart[2])
Vx, Vy, Vz = np.array(Vx), np.array(Vy), np.array(Vz)
# plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# draw unit sphere
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(u.size), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.2)

# plot path (latitude circle)
ax.plot(x, y, z, 'b-', linewidth=1.5, label=f'θ = {theta0:.2f} Path')

# plot transported vectors
scale = 0.5
ax.quiver(x[::2], y[::2], z[::2], 
          scale*Vx[::2], scale*Vy[::2], scale*Vz[::2],
          color='red', label='Parallel Transported Vector',
          arrow_length_ratio=0.15, linewidth=1)

ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
ax.set_title(f'Parallel Transport Along Latitude θ={theta0:.2f}')
ax.legend()
ax.view_init(elev=30, azim=-45)
plt.savefig('Plots/unit_speed_pt_f1', bbox_inches='tight')
plt.show()

# g) Make a plot that measure the strength for different θ_0.

# define range of theta0 values
theta_vals = np.linspace(0, np.pi, 100)  # Vary from 0 to π

alpha, beta = 1.0, 0.0  # initial vector components

inner_products = []

for theta0 in theta_vals:
    # holonomy rotation angle
    delta = 2 * np.pi * (1 - np.cos(theta0))

    # transported vector at ϕ = 2π
    V_theta_final = alpha * np.cos(delta) - beta * np.sin(delta)
    V_phi_final = alpha * np.sin(delta) + beta * np.cos(delta)

    # compute inner product: <V_initial, V_final>
    inner_product = alpha * V_theta_final + beta * V_phi_final
    inner_products.append(inner_product)

# plot inner product vs. θ_0
plt.figure(figsize=(8, 6))
plt.plot(theta_vals, inner_products, 'r-', linewidth=2)
plt.xlabel(r'$\theta_0$ (initial latitude)')
plt.ylabel(r'Inner Product $\langle V_{\mathrm{initial}}, V_{\mathrm{final}} \rangle$')
plt.title('Holonomy Strength vs. Initial Latitude')
plt.grid(True)
plt.savefig('Plots/holonomy_strength_g1', bbox_inches='tight')
plt.show()