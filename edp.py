import numpy as np

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
    d2u_dx2 = np.zeros(len(u))
    d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
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
    d2psi_dx2 = np.zeros(len(psi), dtype=np.complex128)
    d2psi_dx2[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / dx**2
    d2psi_dx2[0] = d2psi_dx2[-1] = 0  # Dirichlet

    dpsi_dt = -1j * (hbar / (2 * m)) * d2psi_dx2 + (-1j / hbar) * V * psi
    return dpsi_dt

def klein_gordon(t, state, params) -> np.ndarray:
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
    phi, dphi_dt = state
    d2phi_dx2 = np.zeros(len(phi))
    d2phi_dx2[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx**2
    d2phi_dx2[0] = d2phi_dx2[-1] = 0  # Dirichlet

    dphi_dt = dphi_dt
    d2phi_dt2 = c**2 * d2phi_dx2 - m**2 * phi

    return np.array([dphi_dt, d2phi_dt2])

def wave_equation(t, state, params) -> np.ndarray:    
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
    d2u_dx2 = np.zeros(len(u))
    d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    d2u_dx2[0] = d2u_dx2[-1] = 0  # Dirichlet

    du_dt = v
    dv_dt = c**2 * d2u_dx2

    dstate_dt = np.array([du_dt, dv_dt])
    return dstate_dt

