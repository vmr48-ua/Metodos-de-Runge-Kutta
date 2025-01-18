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
from schrodinger import schrodinger_total
from klein_gordon import klein_gordon_total
from difusion import diffusion_total
# externas
import numpy as np
from numpy import pi, exp, sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

######################################################################################################33
# SCHRODINGER
######################################################################################################33
L = 15.
Nx = 200
x = np.linspace(0, L, Nx)
dx = L/(Nx-1)     # Paso espacial

T = 10.       # Tiempo total
Nt = int(1e5) # Puntos temporales
Nt_vec = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])

n = 4         # Parametro de la condicion inicial
m = 1.        # Masa de la partícula

params_schr = (1.0, m, dx, np.zeros(Nx))  # Potencial V(x) = 0
u0_schr = sin(n*np.pi*x/L)  + 0j        # Seno con n+1 nodos
schrodinger_total(L, T, Nx, Nt, n, u0_schr, params_schr, Nt_vec, plot=True)

######################################################################################################33
# DIFUSIÓN
######################################################################################################33
L = 15.
Nx = 200
x = np.linspace(0, L, Nx)
dx = L/(Nx-1)     # Paso espacial

T = 10.       # Tiempo total
Nt = int(1e5) # Puntos temporales
Nt_vec = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])

D = 0.03      # Coeficiente de difusión


params_diff = (D, dx)
# SI SE TOCA LA FUNCIÓN DE DIFUSIÓN, CAMBIAR SOLUCIÓN ANALITICA DE difusion.py
u0_diff = (4*pi*D)**(-0.5)*(exp(-((x-L/3)**2)/(4*D)) + 5*exp(-((x-2*L/3)**2)/(4*D)))
diffusion_total(L, T, Nx, Nt, D, u0_diff, params_diff, Nt_vec, anim=True)

######################################################################################################33
# KLEIN -GORDON
######################################################################################################33
L = 15.
Nx = 200
x = np.linspace(0, L, Nx)
dx = L/(Nx-1)     # Paso espacial

T = 10.       # Tiempo total
Nt = int(1e5) # Puntos temporales

c = 1.        # Velocidad de la onda
m = 1.        # Masa del campo

params_kg = (c, m, dx)
phi0_kg = exp(-((x-L/2)**2)/(2*0.5**2)) # Campo gaussiano
u0_kg = np.array([phi0_kg, np.zeros(Nx)])  # Estado inicial kg
klein_gordon_total(L, T, Nx, Nt, u0_kg, params_kg, Nt_vec, anim=True)