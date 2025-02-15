"""
           Autor:
   Juan Pablo Buitrago Rios
   juanybrisagames@gmail.com
   Version 2.0 : 14/02/2025 11:20pm

"""


import numpy as np #Importe necesario para calculos matematicos
import matplotlib.pyplot as plt #Importe necesario para las graficas

# Definir la función g(x) para el método de punto fijo
def g(x):
    # Ejercicio 1: g(x) = (3x-1)^(1/2)
    # return (3*x-1)**(1/2)
    
    # Ejercicio 2: g(x) = e^x / 4
    return np.exp(x)/4
    
    # Ejercicio 3: g(x) = cos(x)
    # return np.cos(x)

# Derivada de g(x) para verificar criterio de convergencia
def g_prime(x):
    # Ejercicio 1: derivada de g(x)
    # return (3/2) * (3*x - 1)**(-1/2)
    
    # Ejercicio 2: derivada de g(x)
    return np.exp(x)/4
    
    # Ejercicio 3: derivada de g(x)
    # return np.sin(-x)
    
# Función para calcular el error absoluto
def error_absoluto(x_new, x_old):
    return abs(x_new - x_old)

# Función para calcular el error relativo
def error_relativo(x_new, x_old):
    return abs((x_new - x_old) / x_new)

# Función para calcular el error cuadrático
def error_cuadratico(x_new, x_old):
    return (x_new - x_old)**2

# Implementación del método de punto fijo
def punto_fijo(x0, tol=1e-5, max_iter=100):
    iteraciones = []  # Almacena datos de cada iteración
    errores_abs = []  # Lista para errores absolutos
    errores_rel = []  # Lista para errores relativos
    errores_cuad = []  # Lista para errores cuadráticos

    x_old = x0  # Valor inicial
    for i in range(max_iter):
        x_new = g(x_old)  # Aplicar función g(x)
        e_abs = error_absoluto(x_new, x_old)  # Calcular error absoluto
        e_rel = error_relativo(x_new, x_old)  # Calcular error relativo
        e_cuad = error_cuadratico(x_new, x_old)  # Calcular error cuadrático

        # Guardar iteración y errores
        iteraciones.append((i+1, x_new, e_abs, e_rel, e_cuad))
        errores_abs.append(e_abs)
        errores_rel.append(e_rel)
        errores_cuad.append(e_cuad)

        # Criterio de parada por tolerancia
        if e_abs < tol:
            break

        x_old = x_new  # Actualizar valor para la siguiente iteración

    return iteraciones, errores_abs, errores_rel, errores_cuad

# Parámetros iniciales
# x0 = 1.5  # Ejercicio 1
x0 = 1.0  # Ejercicio 2
# x0 = 0.5  # Ejercicio 3

# Ejecutar el método de punto fijo
iteraciones, errores_abs, errores_rel, errores_cuad = punto_fijo(x0)

# Imprimir resultados
print("Iteración | x_n      | Error absoluto | Error relativo | Error cuadrático")
print("-----------------------------------------------------------------------")
for it in iteraciones:
    print(f"{it[0]:9d} | {it[1]:.6f} | {it[2]:.6e} | {it[3]:.6e} | {it[4]:.6e}")

# Graficar la función g(x) y la recta y=x
x_vals = np.linspace(-1, 3, 100)
y_vals = g(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label=r"$\frac{e^x}{4}$", color="blue")  # Función g(x)
plt.plot(x_vals, x_vals, linestyle="dashed", color="red", label="y = x")  # Línea de identidad

# Graficar los puntos de iteración
x_points = [it[1] for it in iteraciones]
y_points = [g(x) for x in x_points]
plt.scatter(x_points, y_points, color="black", zorder=3)
plt.plot(x_points, y_points, linestyle="dotted", color="black", label="Iteraciones")

plt.xlabel("x")
plt.ylabel("g(x)")
plt.legend()
plt.grid(True)
plt.title("Método de Punto Fijo")
plt.savefig("punto_fijo_convergencia.png")  # Guardar la imagen
plt.show()

# Graficar la evolución de los errores
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errores_abs) + 1), errores_abs, marker="o", label="Error absoluto")
plt.plot(range(1, len(errores_rel) + 1), errores_rel, marker="s", label="Error relativo")
plt.plot(range(1, len(errores_cuad) + 1), errores_cuad, marker="^", label="Error cuadrático")

plt.xlabel("Iteración")
plt.ylabel("Error")
plt.yscale("log")  # Escala logarítmica para mejor visualización
plt.legend()
plt.grid(True)
plt.title("Evolución de los Errores")
plt.savefig("errores_punto_fijo.png")  # Guardar la imagen
plt.show()
