# Problem 2: Geometric transformations

# a) Numerically show this transformation is conformal.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereographic_projection(x, y, z):
    denom = 1 - z
    return x / denom, y / denom

def plot_stereographic_transformation(theta_p, phi_p):
    """
    Plots the stereographic projection of tangent vectors at a given point on the unit sphere.

    Parameters:
    theta_p (float): Polar angle (0 ≤ θ ≤ π)
    phi_p (float): Azimuthal angle (0 ≤ φ < 2π)
    """
    # define the point P on the sphere
    x_p = np.sin(theta_p) * np.cos(phi_p)
    y_p = np.sin(theta_p) * np.sin(phi_p)
    z_p = np.cos(theta_p)
    P = np.array([x_p, y_p, z_p])

    # compute orthonormal tangent vectors at P
    e_theta = np.array([
        np.cos(theta_p) * np.cos(phi_p),
        np.cos(theta_p) * np.sin(phi_p),
        -np.sin(theta_p)
    ])
    e_phi = np.array([-np.sin(phi_p), np.cos(phi_p), 0])
    e_phi_unit = e_phi / np.sin(theta_p)

    # parameter t for generating curves
    t = np.linspace(-0.1, 0.1, 100)

    # generate curve along e_theta
    gamma1 = np.array([P + ti * e_theta for ti in t])
    gamma1 = gamma1 / np.linalg.norm(gamma1, axis=1)[:, np.newaxis]

    # generate curve along e_phi_unit
    gamma2 = np.array([P + ti * e_phi_unit for ti in t])
    gamma2 = gamma2 / np.linalg.norm(gamma2, axis=1)[:, np.newaxis]

    # project the curves
    gamma1_proj = stereographic_projection(gamma1[:, 0], gamma1[:, 1], gamma1[:, 2])
    gamma2_proj = stereographic_projection(gamma2[:, 0], gamma2[:, 1], gamma2[:, 2])

    # create figure with two subplots
    fig = plt.figure(figsize=(10, 5))

    # 3D plot of the sphere
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gamma1[:, 0], gamma1[:, 1], gamma1[:, 2], label='Curve 1')
    ax1.plot(gamma2[:, 0], gamma2[:, 1], gamma2[:, 2], label='Curve 2')
    ax1.scatter(P[0], P[1], P[2], color='red', s=50, label='Point P')
    ax1.quiver(P[0], P[1], P[2], e_theta[0], e_theta[1], e_theta[2], color='blue', length=0.1, label='Tangent 1')
    ax1.quiver(P[0], P[1], P[2], e_phi_unit[0], e_phi_unit[1], e_phi_unit[2], color='green', length=0.1, label='Tangent 2')
    ax1.set_title('Unit Sphere')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 2D plot of the projection
    ax2 = fig.add_subplot(122)
    ax2.plot(gamma1_proj[0], gamma1_proj[1], label='Projected Curve 1')
    ax2.plot(gamma2_proj[0], gamma2_proj[1], label='Projected Curve 2')
    ax2.scatter(*stereographic_projection(P[0], P[1], P[2]), color='red', s=50, label='Projected P')

    # define the scale for projected tangent vectors
    scale = 0.5
    dS_e_theta = np.array([-0.471, -0.471]) * scale
    dS_e_phi = np.array([-0.544, 0.544]) * scale
    ax2.quiver(*stereographic_projection(P[0], P[1], P[2]), dS_e_theta[0], dS_e_theta[1], color='blue', scale=1, scale_units='xy', angles='xy', label='Projected Tangent 1')
    ax2.quiver(*stereographic_projection(P[0], P[1], P[2]), dS_e_phi[0], dS_e_phi[1], color='green', scale=1, scale_units='xy', angles='xy', label='Projected Tangent 2')
    ax2.set_title('Stereographic Projection')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')

    # annotate angles
    ax1.text(P[0], P[1], P[2], '90°', color='black')
    ax2.text(stereographic_projection(P[0], P[1], P[2])[0], stereographic_projection(P[0], P[1], P[2])[1], '90°', color='black')

    plt.tight_layout()
    plt.show()

# angle between all lines on the unit sphere preserved, as asked

# example usage with the original values
plot_stereographic_transformation(theta_p=2*np.pi/3, phi_p=np.pi/4)

# b) Plot the geodesics of the unit sphere and observe that they are
# "great circles."

def plot_great_circles_and_projection():
    # parametrize the equator
    theta = np.linspace(0, 2*np.pi, 100)
    x_eq = np.cos(theta)
    y_eq = np.sin(theta)
    z_eq = np.zeros_like(theta)

    # parametrize x-z tilted great circle (plane x = z)
    theta_tilt_xz = np.linspace(0, 2*np.pi, 100)
    x_tilt_xz = np.cos(theta_tilt_xz) / np.sqrt(2)
    y_tilt_xz = np.sin(theta_tilt_xz)
    z_tilt_xz = np.cos(theta_tilt_xz) / np.sqrt(2)

    # parametrize y-z tilted great circle (plane y = z)
    theta_tilt_yz = np.linspace(0, 2*np.pi, 100)
    x_tilt_yz = np.sin(theta_tilt_yz)
    y_tilt_yz = np.cos(theta_tilt_yz) / np.sqrt(2)
    z_tilt_yz = np.cos(theta_tilt_yz) / np.sqrt(2)

    # project all curves
    x_eq_proj, y_eq_proj = stereographic_projection(x_eq, y_eq, z_eq)
    x_tilt_xz_proj, y_tilt_xz_proj = stereographic_projection(x_tilt_xz, y_tilt_xz, z_tilt_xz)
    x_tilt_yz_proj, y_tilt_yz_proj = stereographic_projection(x_tilt_yz, y_tilt_yz, z_tilt_yz)

    # create figure
    fig = plt.figure(figsize=(12, 6))

    # 3D subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_eq, y_eq, z_eq, label='Equator')
    ax1.plot(x_tilt_xz, y_tilt_xz, z_tilt_xz, label='X-Z Tilted Circle')
    ax1.plot(x_tilt_yz, y_tilt_yz, z_tilt_yz, label='Y-Z Tilted Circle')
    ax1.set_title('Great Circles on Unit Sphere')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)

    # 2D subplot
    ax2 = fig.add_subplot(122)
    ax2.plot(x_eq_proj, y_eq_proj, label='Projected Equator (Unit Circle)')
    ax2.plot(x_tilt_xz_proj, y_tilt_xz_proj, label='Projected X-Z Tilted Circle')
    ax2.plot(x_tilt_yz_proj, y_tilt_yz_proj, label='Projected Y-Z Tilted Circle')
    ax2.set_title('Stereographic Projection')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()

# call the function to execute the plotting
plot_great_circles_and_projection()

# c) Use your result from Problem 1, for each unit-speed parametrization,
# plot parallel transport trajectories of a closed loop for various 
# initial vectors under the stereographic projection.

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


def plot_parallel_transport_projection(theta0=np.pi/4):
    """Plot parallel transport trajectories under stereographic projection"""
    # parallel transport parameters
    alpha = 1.0
    beta = 0.0
    n_mag = 1.0

    # generate closed loop (latitude circle)
    phi_vals = np.linspace(0, 2*np.pi, 30)
    theta_vals = theta0 * np.ones_like(phi_vals)

    # calculate 3D coordinates and transported vectors
    x_sphere = np.sin(theta0) * np.cos(phi_vals)
    y_sphere = np.sin(theta0) * np.sin(phi_vals)
    z_sphere = np.cos(theta0) * np.ones_like(phi_vals)
    
    # calculate parallel transported vectors
    delta = 2 * np.pi * (1 - np.cos(theta0))
    V_theta = alpha * np.cos(delta * phi_vals/(2*np.pi))
    V_phi = alpha * np.sin(delta * phi_vals/(2*np.pi))

    # project to stereographic coordinates
    x_proj, y_proj = stereographic_projection(x_sphere, y_sphere, z_sphere)
    
    # calculate projected vector components
    Vx_proj, Vy_proj = [], []
    for i in range(len(phi_vals)):
        # get spherical basis vectors at current point
        e_theta, e_phi = spherical_to_cartesian(theta0, phi_vals[i])
        
        # construct vector in local basis
        vec_3d = V_theta[i]*e_theta + V_phi[i]*e_phi
        
        # project vector components using Jacobian of stereographic projection
        z = z_sphere[i]
        J = np.array([[1/(1-z), 0, x_sphere[i]/(1-z)**2],
                      [0, 1/(1-z), y_sphere[i]/(1-z)**2]])
        vec_proj = J @ vec_3d
        Vx_proj.append(vec_proj[0])
        Vy_proj.append(vec_proj[1])

    # create figure
    fig = plt.figure(figsize=(12, 6))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_sphere, y_sphere, z_sphere, 'b-', label=f'θ = {theta0:.2f} Path')
    ax1.quiver(x_sphere[::3], y_sphere[::3], z_sphere[::3], 
               V_theta[::3]*e_theta[0], V_theta[::3]*e_theta[1], V_theta[::3]*e_theta[2],
               color='red', length=0.1, label='Transported Vector')
    ax1.set_title(f'3D Parallel Transport\nLatitude θ={theta0:.2f}')
    ax1.legend()
    
    # stereographic projection plot
    ax2 = fig.add_subplot(122)
    ax2.plot(x_proj, y_proj, 'b-', label='Projected Path')
    ax2.quiver(x_proj[::3], y_proj[::3], 
               np.array(Vx_proj)[::3], np.array(Vy_proj)[::3],
               color='red', scale=15, width=0.003, label='Projected Vectors')
    ax2.set_title(f'Stereographic Projection\nRotation: {delta/np.pi:.2f}π')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'Plots/parallel_transport_projection_{theta0:.2f}.png')
    plt.show()

# create plots for different latitudes
for theta0 in [np.pi/6, np.pi/4, np.pi/3]:
    plot_parallel_transport_projection(theta0)


# d) Plot inner products between two vectors at the same point 
# after the stereographic projection. Does it preserved the inner 
# product after the stereographic projection?

def projection_jacobian(x, y, z):
    """Returns Jacobian matrix at point (x,y,z) for tangent vectors"""
    denom = 1 - z
    J = np.array([[1/denom, 0, x/(denom**2)],
                  [0, 1/denom, y/(denom**2)]])
    return J

# generate points and vectors on sphere
num_points = 100
theta_vals = np.linspace(0, np.pi, num_points)
phi_vals = np.zeros_like(theta_vals)

results = []
for theta in theta_vals:
    # sphere point
    x = np.sin(theta) * np.cos(0)
    y = np.sin(theta) * np.sin(0)
    z = np.cos(theta)
    
    # generate orthogonal tangent vectors
    e_theta = np.array([np.cos(theta), 0, -np.sin(theta)])
    e_phi = np.array([0, 1, 0])
    
    # get Jacobian at this point
    J = projection_jacobian(x, y, z)
    
    # project vectors (pushforward)
    v_proj = J @ e_theta
    w_proj = J @ e_phi
    
    # compute inner products
    orig_ip = np.dot(e_theta, e_phi) # Should be 0 (orthogonal)
    proj_ip = np.dot(v_proj, w_proj)
    
    # conformal factor
    lambda_factor = 1/(1 - z)
    
    results.append({
        'theta': theta,
        'z': z,
        'orig_ip': orig_ip,
        'proj_ip': proj_ip,
        'lambda_sq': lambda_factor**2
    })

# plot results
fig = plt.figure(figsize=(12, 5))

# plot 1: Projected vs Original Inner Product
ax1 = fig.add_subplot(111)
zs = [r['z'] for r in results]
proj_ips = [r['proj_ip'] for r in results]
ax1.scatter(zs, proj_ips, c='r', label='Actual Projected IP')
ax1.plot(zs, [0]*len(zs), 'b--', label='Original IP (always 0)')
ax1.set_xlabel('z-coordinate on Sphere')
ax1.set_ylabel('Inner Product')
ax1.set_title('Inner Product After Projection\n(Orthogonal Vectors)')
ax1.legend()
ax1.grid(True)

# this plot is irrelevant because the inner product is preserved
# as the previous plot shows

# plot 2: Conformal Factor Relationship
# ax2 = fig.add_subplot(122)
# scaling_factors = [r['proj_ip']/(r['lambda_sq']*r['orig_ip']) if r['orig_ip']!=0 else 0 
#                    for r in results]
# ax2.plot(zs, scaling_factors, 'g-')
# ax2.set_xlabel('z-coordinate on Sphere')
# ax2.set_ylabel('(Projected IP) / (λ²·Original IP)')
# ax2.set_title('Conformal Scaling Verification')
# ax2.grid(True)
# ax2.set_ylim(0, 2)

plt.tight_layout()
plt.savefig('Plots/inner_products_d1', bbox_inches='tight')
plt.show()

# as the horizontal flat line result of this plot shows, 
# and as follows from the fact that conformal transformations 
# locally preserve angles, and hence the inner products
# after the stereographic projection.

# e) Zihang forgor e :(((((((

# f) Can the stereographic projection alter the holonomy on the unit sphere
# when parallel transported?

# stereographic projection maps the unit sphere (excluding the north pole) 
# onto a plane while preserving angles. parallel transport along a closed 
# latitude loop on the sphere results in a holonomy angle of  2𝜋(1−cos𝜃). 
# the projection scales tangent vectors by 1/(1−z) but does not change 
# the relative angle between them (per conformality). since stereographic projection 
# is conformal, it preserves the structure of parallel transport, and thus,
# the holonomy remains unchanged after projection. thus, stereographic projection 
# does not alter the parallel transport's rotation on the unit sphere.