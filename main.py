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

############################
# DEFINICIÓN DE PARÁMETROS #
############################

L = 10.0   # Dominio espacial
Nx = 100   # Puntos espaciales
dx = L/Nx  # Paso espacial

T = 5.0    # Tiempo total
Nt = 500   # Puntos temporales
dt = T/Nt  # Paso temporal

c = 1.0    # Velocidad de la onda
m = 1.0    # Masa del campo

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

##################
# EDP A RESOLVER #
##################

plot(x,t)