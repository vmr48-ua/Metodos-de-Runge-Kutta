import numpy as np


def RKIV(r0,drdt,dt,N) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 4
    '''
    r = np.zeros((N,len(r0)))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(r[i])
        k2 = dt*drdt(r[i] + k1/2)
        k3 = dt*drdt(r[i] + k2/2)
        k4 = dt*drdt(r[i] + k3)
        r[i+1] = r[i] + (k1+ 2*k2 + 2*k3 + k4)/6
    
    return r
