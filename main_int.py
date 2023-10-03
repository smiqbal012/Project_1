import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps
import warnings

k = 9 * 10**9  # electrostatic constant

def riemann_sum(f, a, b, N):
    dx = (b - a) / N
    x = np.linspace(a + dx/2, b - dx/2, N)
    return np.sum(f(x)) * dx

def trapezoidal(f, a, b, N):
    x = np.linspace(a, b, N)
    y = f(x)
    return np.trapz(y, x)

def simpson(f, a, b, N):
    if N % 2 == 1:
        raise ValueError("N must be even for Simpson's Rule!")
    dx = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    return dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])

def E_field_distance(lam_func, r, a, b, N, method):
    integrand = lambda x: lam_func(x) * k / (np.sqrt(r**2 + x**2))
    if method == 'riemann':
        return riemann_sum(integrand, a, b, N)
    elif method == 'trapezoidal':
        return trapezoidal(integrand, a, b, N)
    elif method == 'simpson':
        return simpson(integrand, a, b, N)

lam_uniform = lambda x: np.where((x >= a) & (x <= b), 1, 0)
lam_linear = lambda x: np.where((x >= a) & (x <= b), x, 0)

charge_distributions = [lam_uniform, lam_linear]
titles = ["Uniform charge distribution", "Linear charge distribution"]

a, b = -1, 1  # limits of integration
N = 1000  # number of partitions
r_values = np.linspace(0.5, 5, 500)  # points at which electric field is calculated
methods = ['riemann', 'trapezoidal', 'simpson']
labels = ['Riemann Sum', 'Trapezoidal Rule', 'Simpson’s Rule']

for lam_func, title in zip(charge_distributions, titles):
    plt.figure(figsize=(10, 6))
    for method, label in zip(methods, labels):
        E_r = [E_field_distance(lam_func, r, a, b, N, method) for r in r_values]
        plt.plot(r_values, E_r, label=label)
        
    # Handle the integration warning and compute electric field with quad
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_r_trapz_scipy = [quad(lambda x: lam_func(x) * k / np.sqrt(r**2 + x**2), a, b)[0] for r in r_values]
    
    y_vals_simps = [lam_func(x) * k / np.sqrt(r**2 + x**2) for r in r_values for x in np.linspace(a, b, N)]
    y_vals_simps = np.array(y_vals_simps).reshape(len(r_values), N)
    E_r_simps_scipy = [simps(y, np.linspace(a, b, N)) for y in y_vals_simps]
    
    plt.plot(r_values, E_r_trapz_scipy, '--', label="Scipy Trapezoidal (quad)")
    plt.plot(r_values, E_r_simps_scipy, ':', label="Scipy Simpson's (simps)")
    
    plt.title(title)
    plt.xlabel('Distance (r)')
    plt.ylabel('Electric Field (E)')
    plt.legend()
    plt.grid(True)
    plt.show()

