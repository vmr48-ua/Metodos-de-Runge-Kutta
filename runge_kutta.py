import numpy as np




def RKII(r0,t,drdt, p= 1/2) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 2 genérico
    '''
    a1 = 1- p
    a2 = p
    
    c = 1/(2*p)

    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N,len(r0)))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(r[i])
        k2 = dt*drdt(r[i] + c*k1)
        r[i+1] = r[i] + a1*k1 + a2*k2
    
    return r

def RKIV(r0,t,drdt) -> np.ndarray:
    '''
    Función que implementa el método de Runge-Kuta 4
    '''
    dt = t[1]-t[0]
    N = len(t)

    r = np.zeros((N,len(r0)))

    r[0,:] = r0

    for i in range(N-1):
        k1 = dt*drdt(r[i])
        k2 = dt*drdt(r[i] + k1/2)
        k3 = dt*drdt(r[i] + k2/2)
        k4 = dt*drdt(r[i] + k3)
        r[i+1] = r[i] + (k1+ 2*k2 + 2*k3 + k4)/6
    
    return r
