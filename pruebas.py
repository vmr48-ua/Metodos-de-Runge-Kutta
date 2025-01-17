
# locales
from runge_kutta import *
from plot_func import *
from edp import *
from eficiencia import *
# externas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

############################
# DEFINICIÓN DE PARÁMETROS #
############################

L = 5.        # Dominio espacial
Nx = 100      # Puntos espaciales
dx = L/Nx     # Paso espacial

T = 5       # Tiempo total
Nt = int(10000) # Puntos temporales
dt = T/Nt     # Paso temporal

c = 1.        # Velocidad de la onda
m = 1.        # Masa del campo
D = 1.        # Coeficiente de difusión

n = 2         #Parametro de la condicion inicial

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

##################
# EDP A RESOLVER #
##################
# Parámetros de las EDP
params_schr = (1.0, m, dx, np.zeros(Nx))  # Potencial V(x) = 0

# Condiciones iniciales

u0_schr = np.sin(n*np.pi*x/L)  + 0j      # Seno con n+1 nodos

# Resolución de las EDP



################
# SCHRODINGER #
################

sol_RKII_G = RKII_G(u0_schr, t, schrodinger, params_schr)



u_schr_RKII_G  = np.real(sol_RKII_G[:,:])    # Campo phi


u_schr_anal = np.zeros((Nt,Nx))
for i in range(Nt):
    u_schr_anal[i,:] = np.sin(n*np.pi*x[:]/L)*np.cos(-((n**2)*(np.pi**2)*t[i])/(2*L**2))

error2 = error_schr(u_schr_anal, u_schr_RKII_G, dx)

plt.figure()
plt.plot(t, error2, label='error RKII_G')
plt.legend()
plt.show()


# Plots que irán en el archivo plot_func.py



# Schrodinger

fig, ax = plt.subplots()
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(-1.1*np.abs(np.min(u_schr_RKII_G)), 1.1*np.abs(np.max(u_schr_RKII_G)))
ax.set_xlabel("x")
ax.set_ylabel(r"$\phi(x,t)$")
ax.set_title("Evolución de Schrodinger $\\phi(x, t)$")

line_phi_RKII_G,  = ax.plot([], [], label="phi(x,t) - RKII_G")


line_phi_anal,  = ax.plot([], [], label="phi(x,t) - analítica")
time_text = ax.text(0.8*L, 0.8*np.max(u_schr_RKII_G), '', fontsize=12)
ax.legend()

def update(frame):
    line_phi_RKII_G.set_data(x, u_schr_RKII_G[frame])

    line_phi_anal.set_data(x, u_schr_anal[frame])
    time_text.set_text(f"t = {np.round(t[frame],2)}s")
    return line_phi_RKII_G, line_phi_anal, time_text

ani = FuncAnimation(fig, update, frames=range(0,Nt,5), blit=True, interval=1)
plt.show()