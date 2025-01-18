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
from numpy import pi, exp, sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

############################
# DEFINICIÓN DE PARÁMETROS #
############################

L = 25.        # Dominio espacial
Nx = 200      # Puntos espaciales
dx = L/Nx     # Paso espacial

T = 10.       # Tiempo total
Nt = int(5e5) # Puntos temporales
dt = T/Nt     # Paso temporal

c = 1.        # Velocidad de la onda
m = 1.        # Masa del campo
D = 0.07      # Coeficiente de difusión

n = 4         # Parametro de la condicion inicial

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

##################
# EDP A RESOLVER #
##################
# Parámetros de las EDP
params_diff = (D, dx)
params_kg = (c, m, dx)
params_schr = (1.0, m, dx, np.zeros(Nx))  # Potencial V(x) = 0
params_wave = (c, dx)

# Condiciones de estabilidad
estabilidad_diff = D*dt/dx**2
...
...
...

# Condiciones iniciales
u0_diff = 1000*(4*pi*D)**(-0.5)*(exp(-((x-L/3)**2)/(4*D)) + 5*exp(-((x-2*L/3)**2)/(4*D))) # Gausianas 
u0_schr = sin(n*np.pi*x/L)  + 0j        # Seno con n+1 nodos
...
phi0_kg = exp(-((x-L/2)**2)/(2*0.5**2)) # Campo gaussiano
u0_kg = np.array([phi0_kg, np.zeros(Nx)])  # Estado inicial kg

# Resolución de las EDP
############
# DIFUSIÓN #
############
if estabilidad_diff >= 0.5:
    raise Exception('Condición de estabilidad no satisfecha')
u_diff_RKII_G, t_RKII_G = timear(RKII_G,(u0_diff, t, diffusion, params_diff))
u_diff_RKIII_G, t_RKIII_G = timear(RKIII_G,(u0_diff, t, diffusion, params_diff))
u_diff_RKIV, t_RKIV = timear(RKIV,(u0_diff, t, diffusion, params_diff))
u_diff_RKVI, t_RKVI = timear(RKVI,(u0_diff, t, diffusion, params_diff))

u_diff_anal = np.zeros((Nt,Nx))
for i in range(Nt):
    u_diff_anal[i,:] = 1000*(4*pi*(t[i]+1)*D)**(-0.5)*(exp(-((x-L/3)**2)/(4*D*(t[i]+1))) + 5*exp(-((x-2*L/3)**2)/(4*D*(t[i]+1))))
    
error2 = error(u_diff_anal, u_diff_RKII_G, dx)[1:]    
error3 = error(u_diff_anal, u_diff_RKIII_G, dx)[1:]
error4 = error(u_diff_anal, u_diff_RKIV, dx)[1:]
error6 = error(u_diff_anal, u_diff_RKVI, dx)[1:]

error2_max = np.max(error2)
error3_max = np.max(error3)
error4_max = np.max(error4)
error6_max = np.max(error6)

fig, ax = plt.subplots()
ax.plot(t[1:], error2, label='RKII_G')
ax.plot(t[1:], error3, label='RKIII_G')
ax.plot(t[1:], error4, label='RKIV')
ax.plot(t[1:], error6, label='RKVI')
ax.set_xlabel('t')
ax.semilogx(), ax.semilogy()
ax.set_ylim(0.8*np.min(error6), 1.2*np.max(error2))
ax.set_ylabel('Error')
ax.set_title('Error de la solución numérica frente a la analítica')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(t_RKII_G, error2_max, label='RKII_G', marker='x')
ax.scatter(t_RKIII_G, error3_max, label='RKIII_G', marker='x')
ax.scatter(t_RKIV, error4_max, label='RKIV', marker='x')
ax.scatter(t_RKVI, error6_max, label='RKVI', marker='x')
ax.set_xlabel('Tiempo de ejecución')
ax.set_ylabel('Error máximo')
ax.semilogx(), ax.semilogy()
ax.set_title('Error de la solución numérica frente a la analítica')
ax.legend()
plt.show()

################
# KLEIN-GORDON #
################
# sol_RKIV = RKIV(u0_kg, t, klein_gordon, params_kg)
# u_kg_RKIV  = sol_RKIV[:,0,:] # Campo phi
# du_kg_RKIV = sol_RKIV[:,1,:] # Velocidad dphi/dt



################
# SCHRODINGER #
################
# sol_RKVI = RKVI(u0_schr, t, schrodinger, params_schr)
# u_schr_RKVI  = np.real(sol_RKVI[:,:])   # Campo phi

# u_schr_anal = np.zeros((Nt,Nx))
# for i in range(Nt):
#     u_schr_anal[i,:] = np.sin(n*np.pi*x/L)*np.cos(-((n**2)*(np.pi**2)*t[i])/(2*L**2))
# error6 = error(u_schr_anal, u_schr_RKVI, dx)


# Plots que irán en el archivo plot_func.py

# Difusión
fig, ax = plt.subplots(figsize=(7, 7))
plt.tight_layout()
line_RKII_G, = ax.plot(x, u_diff_RKII_G[0], label='RKIIG')
line_RKIII_G, = ax.plot(x, u_diff_RKIII_G[0], label='RKIIIG')
line_RKIV, = ax.plot(x, u_diff_RKIV[0], label='RKIV')
line_RKVI, = ax.plot(x, u_diff_RKVI[0], label='RKVI')
line_anal, = ax.plot(x, u_diff_anal[0], label='Analítica')
ax.set_xlim(np.min(x), np.max(x))
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.set_title('Evolución temporal de la Ecuación de Difusión')
ax.legend()

def update(frame):
    line_RKII_G.set_ydata(u_diff_RKII_G[frame])
    line_RKIII_G.set_ydata(u_diff_RKIII_G[frame])
    line_RKIV.set_ydata(u_diff_RKIV[frame])
    line_RKVI.set_ydata(u_diff_RKVI[frame])
    line_anal.set_ydata(u_diff_anal[frame])
    return line_RKII_G, line_RKIII_G, line_RKIV, line_RKVI, line_anal

ani = FuncAnimation(fig, update, frames=range(0,Nt,100), blit=True, interval=1)
plt.show()



# Klein-Gordon
# fig, ax = plt.subplots()
# ax.set_xlim(np.min(x), np.max(x))
# ax.set_ylim(-1.1*np.abs(np.min(du_kg_RKIV)), 1.1*np.abs(np.max(du_kg_RKIV)))
# ax.set_xlabel("x")
# ax.set_ylabel(r"$\phi(x,t)$")
# ax.set_title("Evolución temporal del campo $\\phi(x, t)$")

# line_phi_RKIV,  = ax.plot([], [], label="phi(x,t) - RKIV")
# line_dphi_RKIV, = ax.plot([], [], label="dphi(x,t)/dt - RKIV")
# time_text = ax.text(0.8*L, 0.8*np.max(u_kg_RKIV), '', fontsize=12)
# ax.legend()

# def update(frame):
#     line_phi_RKIV.set_data(x, u_kg_RKIV[frame])
#     line_dphi_RKIV.set_data(x, du_kg_RKIV[frame])
#     time_text.set_text(f"t = {np.round(t[frame],2)}s")
#     return line_phi_RKIV, line_dphi_RKIV, time_text

# ani = FuncAnimation(fig, update, frames=range(0,Nt,5), blit=True, interval=1)
# plt.show()



# Schrodinger

# fig, ax = plt.subplots()
# ax1 = ax.twinx()
# ax.set_xlim(np.min(x), np.max(x))
# ax.set_ylim(-1.1*np.abs(np.min(u_schr_RKVI)), 1.1*np.abs(np.max(u_schr_RKVI)))
# ax.set_xlabel("x")
# ax.set_ylabel(r"$\phi(x,t)$")
# ax.set_title("Evolución de Schrodinger $\\phi(x, t)$")
# ax1.plot(t, error6, label='error RKVI')
# ax1.set_ylabel('Error')
# ax1.legend()

# line_phi_RKVI,  = ax.plot([], [], label="phi(x,t) - RKVI")
# line_phi_anal,  = ax.plot([], [], label="phi(x,t) - analítica")
# time_text = ax.text(0.8*L, 0.8*np.max(u_schr_RKVI), '', fontsize=12)
# ax.legend()

# def update(frame):
#     line_phi_RKVI.set_data(x, u_schr_RKVI[frame])
#     line_phi_anal.set_data(x, u_schr_anal[frame])
#     time_text.set_text(f"t = {np.round(t[frame],2)}s")
#     return line_phi_RKVI, line_phi_anal, time_text

# ani = FuncAnimation(fig, update, frames=range(0,Nt,25), blit=True, interval=1)
# plt.show()