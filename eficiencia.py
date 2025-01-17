import matplotlib.pyplot as plt
import numpy as np
import time
import numpy.linalg as la
"""
Vamos a calcular el error de la solución analítica de 
la ecuación de Schrödinger frente a la solución numérica
que proporcionan los distintos métodos de Runge-Kutta.
"""

def error_schr(psi_analitica, psi_numerica, dx):
    return la.norm(psi_analitica - psi_numerica,axis=1)*dx

def timear(funcion, param):
    """ 
    Devuelve el tiempo que tarda en ejecutarse una función
    """
    t0 = time.perf_counter()
    funcion(param)
    t1 = time.perf_counter()
    return(np.abs(t1-t0))

def plot_eficiencia():
    fig1,ax1 = plt.subplots()
    ax1.set_title('Error de la parte real de $\\Psi (x)$\nfrente al tiempo de ejecución')
    ax1.set_xlabel('Tiempo de ejecución')
    ax1.set_ylabel('Error')
    
    fig2,ax2 = plt.subplots()
    ax2.set_title('Error máximo de la parte real de $\\Psi (x)$\nfrente a tiempo de computación')
    ax1.set_xlabel('Tiempo de computación')
    ax1.set_ylabel('Error máximo')

    return fig1, fig2