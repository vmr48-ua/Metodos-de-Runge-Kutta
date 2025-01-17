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
from edp import *
# externas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

############################
# DEFINICIÓN DE PARÁMETROS #
############################

L = 5.        # Dominio espacial
Nx = 150      # Puntos espaciales
dx = L/Nx     # Paso espacial

T = 5.        # Tiempo total
Nt = int(2e4) # Puntos temporales
dt = T/Nt     # Paso temporal

c = 2.        # Velocidad de la onda
m = 3.        # Masa del campo
D = 1.        # Coeficiente de difusión

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

##################
# EDP A RESOLVER #
##################
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
phi0_kg = np.exp(-((x-L/2)**2)/(2*0.5**2)) # Campo gaussiano
u0_kg = np.array([phi0_kg, np.zeros(Nx)])  # Estado inicial kg

# Resolución de las EDP
############
# DIFUSIÓN #
############
if estabilidad_diff >= 0.5:
        raise Exception('Condición de estabilidad no satisfecha')
u_diff_RKII_G = RKII_G(u0_diff, t, diffusion, params_diff)
u_diff_RKIV = RKIV(u0_diff, t, diffusion, params_diff)

################
# KLEIN-GORDON #
################
sol_RKIV = RKIV(u0_kg, t, klein_gordon, params_kg)
u_kg_RKIV  = sol_RKIV[:,0,:] # Campo phi
du_kg_RKIV = sol_RKIV[:,1,:] # Velocidad dphi/dt

# Plots que irán en el archivo plot_func.py
# Difusión
fig, ax = plt.subplots(figsize=(7, 7))
plt.tight_layout()
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

ani = FuncAnimation(fig, update, frames=range(0,Nt,5), blit=True, interval=1)
plt.show()



# Klein-Gordon
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-1.1*np.abs(np.min(du_kg_RKIV)), 1.1*np.abs(np.max(du_kg_RKIV)))
ax.set_xlabel("x")
ax.set_ylabel(r"$\phi(x,t)$")
ax.set_title("Evolución temporal del campo $\\phi(x, t)$")

line_phi_RKIV,  = ax.plot([], [], label="phi(x,t) - RKIV")
line_dphi_RKIV, = ax.plot([], [], label="dphi(x,t)/dt - RKIV")
time_text = ax.text(0.8*L, 0.8*np.max(u_kg_RKIV), '', fontsize=12)
ax.legend()

def update(frame):
    line_phi_RKIV.set_data(x, u_kg_RKIV[frame])
    line_dphi_RKIV.set_data(x, du_kg_RKIV[frame])
    time_text.set_text(f"t = {np.round(t[frame],2)}s")
    return line_phi_RKIV, line_dphi_RKIV, time_text

ani = FuncAnimation(fig, update, frames=range(0,Nt,5), blit=True, interval=1)
plt.show()