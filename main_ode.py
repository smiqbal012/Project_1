import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
L = 1.0
C = 1.0
R_values = [0.5, 1.0, 2.0]
initial_conditions = [1.0, 0.0]
t = np.linspace(0, 30, 1000)

# RLC system
def rlc_circuit(t, y, L, R, C):
    q, i = y
    dqdt = i
    didt = (-R*i - q/C)/L
    return [dqdt, didt]

# Analytical Solution
def analytic_solution(t, L, R, C):
    omega0 = 1/np.sqrt(L*C)
    if R == 1/np.sqrt(L/C):
        # Critically damped
        q = initial_conditions[0] * (1 + (1/np.sqrt(L*C))*t) * np.exp(-t/(2*L*C))
        i = initial_conditions[0] * (1 + (1/np.sqrt(L*C))*(t-1)) * np.exp(-t/(2*L*C))
    elif R < 1/np.sqrt(L/C):
        # Underdamped
        gamma = R/(2*L)
        omega = np.sqrt(omega0**2 - gamma**2)
        q = initial_conditions[0] * np.exp(-gamma*t) * np.cos(omega*t)
        i = -initial_conditions[0] * np.exp(-gamma*t) * (gamma*np.cos(omega*t) + omega*np.sin(omega*t))
    else:
        # Overdamped
        gamma = R/(2*L)
        omega = np.sqrt(gamma**2 - omega0**2)
        q = initial_conditions[0] * (np.exp(-gamma*t) * (np.cosh(omega*t) + (gamma/omega)*np.sinh(omega*t)))
        i = -initial_conditions[0] * np.exp(-gamma*t) * (gamma*np.cosh(omega*t) + omega*np.sinh(omega*t))
    return q, i

# Forward Euler method
def forward_euler(func, y0, t, args):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        y[i+1] = y[i] + (t[i+1] - t[i]) * np.array(func(t[i], y[i], *args))
    return y

# 4th order Runge-Kutta method
def rk4(func, y0, t, args):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n-1):
        dt = t[i+1] - t[i]
        k1 = np.array(func(t[i], y[i], *args))
        k2 = np.array(func(t[i] + 0.5*dt, y[i] + 0.5*dt*k1, *args))
        k3 = np.array(func(t[i] + 0.5*dt, y[i] + 0.5*dt*k2, *args))
        k4 = np.array(func(t[i] + dt, y[i] + dt*k3, *args))
        y[i+1] = y[i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y

# Solving and Plotting
for R in R_values:
    # Analytical solution
    q_analytic, i_analytic = analytic_solution(t, L, R, C)
    
    # Forward Euler
    y_fe = forward_euler(rlc_circuit, initial_conditions, t, args=(L, R, C))
    
    # 4th order Runge-Kutta
    y_rk4 = rk4(rlc_circuit, initial_conditions, t, args=(L, R, C))
    
    # SciPy's solve_ivp
    sol = solve_ivp(rlc_circuit, [t[0], t[-1]], initial_conditions, t_eval=t, args=(L, R, C))
    
    # Plotting Charge vs Time
    plt.figure()
    plt.plot(t, q_analytic, 'g', label='Analytical Solution')
    plt.plot(t, y_fe[:, 0], 'b--', label='Forward Euler')
    plt.plot(t, y_rk4[:, 0], 'c:', label='RK4')
    plt.plot(sol.t, sol.y[0], 'r-', label='SciPy solve_ivp')
    plt.title(f"Charge vs Time for R={R}")
    plt.xlabel('Time(t)')
    plt.ylabel('Charge(q)')
    plt.legend()
    
    # Plotting Current vs Time
    plt.figure()
    plt.plot(t, i_analytic, 'g', label='Analytical Solution')
    plt.plot(t, y_fe[:, 1], 'b--', label='Forward Euler')
    plt.plot(t, y_rk4[:, 1], 'c:', label='RK4')
    plt.plot(sol.t, sol.y[1], 'r-', label='SciPy solve_ivp')
    plt.title(f"Current vs Time for R={R}")
    plt.xlabel('Time(t)')
    plt.ylabel('Current(I)')
    plt.legend()

plt.show()


# Validation for SHO behavior
R_sho = 0  # Resistance for SHO
q_sho, i_sho = analytic_solution(t, L, R_sho, C)
y_rk4_sho = rk4(rlc_circuit, initial_conditions, t, args=(L, R_sho, C))

# Plotting Charge vs Time for SHO
plt.figure()
plt.plot(t, q_sho, 'g', label='SHO Analytical Solution')
plt.plot(t, y_rk4_sho[:, 0], 'c:', label='RK4')
plt.title(f"Charge vs Time for Simple Harmonic Oscillator (R={R_sho})")
plt.xlabel('Time')
plt.ylabel('Charge')
plt.legend()

# Plotting Current vs Time for SHO
plt.figure()
plt.plot(t, i_sho, 'g', label='SHO Analytical Solution')
plt.plot(t, y_rk4_sho[:, 1], 'c:', label='RK4')
plt.title(f"Current vs Time for Simple Harmonic Oscillator (R={R_sho})")
plt.xlabel('Time(t)')
plt.ylabel('Current(I)')
plt.legend()

plt.show()

