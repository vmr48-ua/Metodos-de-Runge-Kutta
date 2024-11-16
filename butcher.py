import sympy as sp

def generate_runge_kutta_coefficients_with_fixed_c(order, fixed_c_values=None):
    """
    Genera coeficientes de un método Runge-Kutta explícito con la posibilidad de imponer valores de c_i previamente.
    """
    if order < 1:
        raise ValueError("El orden debe ser un entero positivo.")

    s = order  # Usamos el número mínimo de etapas
    A = sp.Matrix(s, s, lambda i, j: sp.Symbol(f'a{i+1}{j+1}') if j < i else 0)
    b = sp.Matrix(s, 1, lambda i, _: sp.Symbol(f'b{i+1}'))
    c = sp.Matrix(s, 1, lambda i, _: sp.Symbol(f'c{i+1}'))
    c[0] = 0  # Fijamos c1 = 0

    # Imponer valores de c_i si se proporcionan
    if fixed_c_values:
        for i, value in enumerate(fixed_c_values):
            if i < s:
                c[i] = value

    # Ecuaciones para el sistema
    equations = [sp.Eq(sum(b[i, 0] for i in range(s)), 1)]  # Condición de orden 1

    # Condiciones de orden superiores
    for p in range(2, order + 1):
        equations.append(sp.Eq(
            sum(b[i, 0] * sum(A[i, j] * c[j, 0]**(p - 1) for j in range(i)) for i in range(s)),
            1 / p
        ))

    # Consistencia de c_i con A (c_i = sum_j a_ij)
    for i in range(1, s):
        equations.append(sp.Eq(c[i, 0], sum(A[i, j] for j in range(s))))

    # Resolver el sistema de ecuaciones
    unknowns = [A[i, j] for i in range(s) for j in range(i)] + [b[i, 0] for i in range(s)] + [c[i, 0] for i in range(1, s) if c[i] != 0]
    solution = sp.solve(equations, unknowns, dict=True)

    if not solution:
        raise ValueError(f"No se pudo encontrar una solución para un método de orden {order} con c_i fijados")

    solution = solution[0]  # Usar la primera solución encontrada
    A = A.subs(solution)
    b = b.subs(solution)
    c = c.subs(solution)

    return A, b, c

# Solicitar valores de c al usuario e implementar para orden 3
fixed_c_values = [0, 1/2, 1]  # Ejemplo: c1=0, c2=2/3, c3 libre
A_fixed_c, b_fixed_c, c_fixed_c = generate_runge_kutta_coefficients_with_fixed_c(3, fixed_c_values)
print(A_fixed_c)
print(b_fixed_c)
print(c_fixed_c)
