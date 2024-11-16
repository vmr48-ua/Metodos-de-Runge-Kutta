def plot(x0: list,x1: list) -> None:
    """
    Función que plotea cosas
    """
    # Imports
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, axes = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    
    def init():
        for line_real, line_abs, line_comp in zip(lines_real, lines_abs, lines_comp): 
            line_real.set_data([], [])
            line_comp.set_data([], [])
            line_abs.set_data([], [])
        for time_text in time_texts:
            time_text.set_text('')
        return lines_real + lines_comp + lines_abs + time_texts

    def animate(i):
        for idx, (psi, _, _) in enumerate(resultados):
            lines_real[idx].set_data(x, np.real(psi[:, i]))
            lines_comp[idx].set_data(x, np.imag(psi[:, i]))
            lines_abs[idx].set_data(x, np.abs(psi[:, i])**2)
            time_texts[idx].set_text(f't = {i*dt:.2f}s')
        return lines_real + lines_comp + lines_abs + time_texts
        
    anim = FuncAnimation(fig, animate, init_func=init, frames=range(0,Nt,10), interval=1, blit=True)
    
    plt.show()