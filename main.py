import runge_kutta

import numpy as np
import matplotlib.pyplot as plt

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