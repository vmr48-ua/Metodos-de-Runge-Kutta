import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp, sin, cos
from matplotlib.animation import FuncAnimation

# Locales
from runge_kutta import *
from edp import *
from plot_func import *


def klein_gordon_total(L, T, Nx, Nt, u0_kg, params_kg, Nt_vec, anim=True):
    def animation():
        fig, ax = plt.subplots()
        ax.set_xlim(np.min(x), np.max(x))
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

    x = np.linspace(0, L, Nx)
    dx = L/(Nx-1)

    t = np.linspace(0, T, Nt)
    dt = T/(Nt-1)
    
    sol_RKIV = RKIV(u0_kg, t, klein_gordon, params_kg)
    u_kg_RKIV  = sol_RKIV[:,0,:] # Campo phi
    du_kg_RKIV = sol_RKIV[:,1,:] # Velocidad dphi/dt

    if anim:
        animation()