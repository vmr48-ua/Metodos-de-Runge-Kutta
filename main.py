'''
PRESENTACIÓN FÍSICA COMPUTACIONAL - BLOQUE I
INTRODUCCIÓN A LA MODELIZACIÓN EN FÍSICA

MÉTODOS DE RUNGE KUTTA

Víctor Mira Ramírez
74528754Z
vmr48@alu.ua.es

Marco Mas Sempere
74392068V
mmsm13@alu.ua.es
'''
############################
# IMPORTACIÓN DE LIBRERÍAS #
############################

# locales
from runge_kutta import *
from plot_func import *
# externas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

############################
# DEFINICIÓN DE PARÁMETROS #
############################

L = 5.0       # Dominio espacial
Nx = 150      # Puntos espaciales
dx = L/Nx     # Paso espacial

T = 2.0       # Tiempo total
Nt = int(2e4) # Puntos temporales
dt = T/Nt     # Paso temporal

c = 1.0       # Velocidad de la onda
m = 1.0       # Masa del campo
D = 1.0       # Coeficiente de difusión

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

##################
# EDP A RESOLVER #
##################
def diffusion(t, u, params) -> np.ndarray:
    """
    Derivada temporal para la ecuación de difusión.
    
    Input:
    - t: tiempo actual (no se usa en este caso)
    - u: np.ndarray, distribución de la cantidad a difundir en el tiempo t
    - params: tuple (D, dx)
    
    Output:
    - du_dt: np.ndarray, derivada de u con respecto al tiempo
    """
    D, dx = params
    d2u_dx2 = np.zeros_like(u)
    for i in range(1, len(u) - 1):
        d2u_dx2[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
    d2u_dx2[0] = d2u_dx2[-1] = 0  # Dirichlet

    du_dt = D * d2u_dx2
    return du_dt

def schrodinger(t, psi, params) -> np.ndarray:
    """
    Derivada temporal para la ecuación de Schrödinger.
    
    Input:
    - t: tiempo actual (no se usa porque es psi independiente de t)
    - psi: np.ndarray, función de onda en el tiempo t
    - params: tuple (hbar, m, dx, V)
    
    Output:
    - dpsi_dt: np.ndarray, derivada de psi con respecto al tiempo
    """
    hbar, m, dx, V = params
    d2psi_dx2 = np.zeros_like(psi)
    for i in range(1, len(psi) - 1):
        d2psi_dx2[i] = (psi[i + 1] - 2 * psi[i] + psi[i - 1]) / dx**2
    d2psi_dx2[0] = d2psi_dx2[-1] = 0  # Dirichlet

    dpsi_dt = -1j * (hbar / (2 * m)) * d2psi_dx2 + (-1j / hbar) * V * psi
    return dpsi_dt

def klein_gordon(t, phi, params) -> np.ndarray:
    """
    Derivada temporal para la ecuación de Klein-Gordon.
    
    Input:
    - t: tiempo actual (no se usa en este caso)
    - phi: np.ndarray, campo en el tiempo t
    - params: tuple (c, m, dx)
    
    Output:
    - dphi_dt: np.ndarray, derivada de phi con respecto al tiempo
    """
    c, m, dx = params
    d2phi_dx2 = np.zeros_like(phi)
    for i in range(1, len(phi) - 1):
        d2phi_dx2[i] = (phi[i + 1] - 2 * phi[i] + phi[i - 1]) / dx**2
    d2phi_dx2[0] = d2phi_dx2[-1] = 0  # Dirichlet

    dphi_dt = c**2 * d2phi_dx2 - m**2 * phi
    return dphi_dt

def wave_equation(t, state, params):    
    """
    Derivada temporal para la ecuación de onda.
    
    Input:
    - t: tiempo actual (no se usa en este caso)
    - state: np.ndarray, [u, v] u es la posición, v es la velocidad
    - params: tuple (c, dx)
    
    Output:
    - dstate_dt: np.ndarray, [du_dt, dv_dt]
    """
    c, dx = params
    u, v = state
    d2u_dx2 = np.zeros_like(u)
    for i in range(1, len(u) - 1):
        d2u_dx2[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
    d2u_dx2[0] = d2u_dx2[-1] = 0  # Dirichlet

    du_dt = v
    dv_dt = c**2 * d2u_dx2

    dstate_dt = np.array([du_dt, dv_dt])
    return dstate_dt

# Parámetros de las EDP
params_diff = (D, dx)
params_kg = (c, m, dx)
params_sch = (1.0, m, dx, np.zeros(Nx))  # Potencial V(x) = 0
params_wave = (c, dx)

# Condiciones de estabilidad
estabilidad_diff = D*dt/dx**2
...
...
...

# Condiciones iniciales
u0_diff = np.exp(-((x-L/2)**2)/(2*0.5**2)) # Gaussiana centrada en L/2
u0_schr = ...
...
...

# Resolución de las EDP

############
# DIFUSIÓN #
############
if estabilidad_diff >= 0.5:
        raise Exception('Condición de estabilidad no satisfecha, prueba a bajar dt')
u_diff_RKII_G = RKII_G(u0_diff, t, diffusion, params_diff)
u_diff_RKIV = RKIV(u0_diff, t, diffusion, params_diff)



# Plots que irán en el archivo plot_func.py

fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
ax.set_aspect('equal', 'box')
line_RKII_G, = ax.plot(x, u_diff_RKII_G[0], label='RKIIG')
line_RKIV, = ax.plot(x, u_diff_RKIV[0], label='RKIV')
ax.set_xlim(0, L)
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.set_title('Evolución temporal de la Ecuación de Difusión')
ax.legend()

def update(frame):
    line_RKII_G.set_ydata(u_diff_RKII_G[frame])
    line_RKIV.set_ydata(u_diff_RKIV[frame])
    return line_RKII_G, line_RKIV

ani = FuncAnimation(fig, update, frames=range(0, Nt, 1), blit=True, interval=1)

plt.show()