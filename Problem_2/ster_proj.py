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
    # Define the point P on the sphere
    x_p = np.sin(theta_p) * np.cos(phi_p)
    y_p = np.sin(theta_p) * np.sin(phi_p)
    z_p = np.cos(theta_p)
    P = np.array([x_p, y_p, z_p])

    # Compute orthonormal tangent vectors at P
    e_theta = np.array([
        np.cos(theta_p) * np.cos(phi_p),
        np.cos(theta_p) * np.sin(phi_p),
        -np.sin(theta_p)
    ])
    e_phi = np.array([-np.sin(phi_p), np.cos(phi_p), 0])
    e_phi_unit = e_phi / np.sin(theta_p)

    # Parameter t for generating curves
    t = np.linspace(-0.1, 0.1, 100)

    # Generate curve along e_theta
    gamma1 = np.array([P + ti * e_theta for ti in t])
    gamma1 = gamma1 / np.linalg.norm(gamma1, axis=1)[:, np.newaxis]

    # Generate curve along e_phi_unit
    gamma2 = np.array([P + ti * e_phi_unit for ti in t])
    gamma2 = gamma2 / np.linalg.norm(gamma2, axis=1)[:, np.newaxis]

    # Project the curves
    gamma1_proj = stereographic_projection(gamma1[:, 0], gamma1[:, 1], gamma1[:, 2])
    gamma2_proj = stereographic_projection(gamma2[:, 0], gamma2[:, 1], gamma2[:, 2])

    # Create figure with two subplots
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

    # Define the scale for projected tangent vectors
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

    # Annotate angles
    ax1.text(P[0], P[1], P[2], '90°', color='black')
    ax2.text(stereographic_projection(P[0], P[1], P[2])[0], stereographic_projection(P[0], P[1], P[2])[1], '90°', color='black')

    plt.tight_layout()
    plt.show()

# Example usage with the original values
plot_stereographic_transformation(theta_p=2*np.pi/3, phi_p=np.pi/4)

# b) 

def plot_great_circles_and_projection():
    # Parametrize the equator
    theta = np.linspace(0, 2*np.pi, 100)
    x_eq = np.cos(theta)
    y_eq = np.sin(theta)
    z_eq = np.zeros_like(theta)

    # Parametrize x-z tilted great circle (plane x = z)
    theta_tilt_xz = np.linspace(0, 2*np.pi, 100)
    x_tilt_xz = np.cos(theta_tilt_xz) / np.sqrt(2)
    y_tilt_xz = np.sin(theta_tilt_xz)
    z_tilt_xz = np.cos(theta_tilt_xz) / np.sqrt(2)

    # Parametrize y-z tilted great circle (plane y = z)
    theta_tilt_yz = np.linspace(0, 2*np.pi, 100)
    x_tilt_yz = np.sin(theta_tilt_yz)
    y_tilt_yz = np.cos(theta_tilt_yz) / np.sqrt(2)
    z_tilt_yz = np.cos(theta_tilt_yz) / np.sqrt(2)

    # Project all curves
    x_eq_proj, y_eq_proj = stereographic_projection(x_eq, y_eq, z_eq)
    x_tilt_xz_proj, y_tilt_xz_proj = stereographic_projection(x_tilt_xz, y_tilt_xz, z_tilt_xz)
    x_tilt_yz_proj, y_tilt_yz_proj = stereographic_projection(x_tilt_yz, y_tilt_yz, z_tilt_yz)

    # Create figure
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

# Call the function to execute the plotting
plot_great_circles_and_projection()

# c) WHAT THE ACTUAL FUCK AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA