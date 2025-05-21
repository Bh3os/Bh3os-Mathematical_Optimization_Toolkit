import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# ----------- Helper plotting functions -----------

def plot_2d_function(f, x_range, x_min=None):
    x = np.linspace(*x_range, 400)
    y = f(x)
    plt.plot(x, y, label="f(x)")
    if x_min is not None:
        plt.plot(x_min, f(x_min), 'ro', label=f"Min: x={x_min:.3f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.title("2D Line Plot")
    plt.show()

def plot_contour_and_surface(f, bounds, optimum=None, constraint_func=None):
    x = np.linspace(*bounds[0], 100)
    y = np.linspace(*bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = f(np.array([X, Y]))

    # Contour Plot
    plt.contour(X, Y, Z, levels=30, cmap='viridis')
    if constraint_func:
        plt.contour(X, Y, constraint_func(np.array([X, Y])), levels=[0], colors='red', linewidths=2, label="Constraint")
    if optimum is not None:
        plt.plot(optimum[0], optimum[1], 'ro', label="Optimum")
    plt.title("Contour Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    # Surface Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
    if optimum is not None:
        ax.scatter(*optimum, f(optimum), color='red', s=50)
    ax.set_title("3D Surface Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    plt.show()

# ----------- Numerical Methods -----------

# Steepest Descent (Gradient Descent)
def steepest_descent(f, grad_f, x0, lr=0.1, max_iter=100):
    x = x0.copy()
    for i in range(max_iter):
        x -= lr * grad_f(x)
    return x

# Newton's Method
def newtons_method(f, grad_f, hess_f, x0, max_iter=10):
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        x -= np.linalg.solve(hess, grad)
    return x

# Penalty Method (Exterior)
def exterior_penalty_method(f, constraint, x0, r=10, max_iter=5):
    def penalized_function(x):
        return f(x) + r * max(0, constraint(x))**2

    result = minimize(penalized_function, x0)
    return result.x

# Log Barrier Method (Interior)
def log_barrier_method(f, constraint, x0, t=1, mu=10, max_iter=10):
    x = x0.copy()
    for i in range(max_iter):
        def barrier_obj(x):
            return t * f(x) - np.log(-constraint(x))

        res = minimize(barrier_obj, x)
        x = res.x
        t *= mu
    return x

# ----------- Example Usages -----------

if __name__ == "__main__":
    # Single-variable: f(x) = (x - 2)^2
    f1 = lambda x: (x - 2)**2
    df1 = lambda x: 2*(x - 2)
    x_min = steepest_descent(f1, df1, x0=np.array([5.0]), lr=0.1)
    plot_2d_function(f1, (0, 5), x_min)

    # Multivariable: f(x, y) = (x - 1)^2 + (y + 2)^2
    f2 = lambda v: (v[0] - 1)**2 + (v[1] + 2)**2
    df2 = lambda v: np.array([2*(v[0] - 1), 2*(v[1] + 2)])
    hess2 = lambda v: np.array([[2, 0], [0, 2]])
    x0 = np.array([3.0, 3.0])
    min_point = newtons_method(f2, df2, hess2, x0)
    plot_contour_and_surface(f2, bounds=[(-5, 5), (-5, 5)], optimum=min_point)

    # Constrained Optimization using Penalty
    # f(x, y) = x^2 + y^2, g(x, y) = x + y - 1 >= 0
    f3 = lambda v: v[0]**2 + v[1]**2
    g3 = lambda v: 1 - (v[0] + v[1])  # constraint g(x) >= 0
    penalty_solution = exterior_penalty_method(f3, g3, x0=np.array([2.0, 2.0]))
    plot_contour_and_surface(f3, bounds=[(-1, 2), (-1, 2)], optimum=penalty_solution, constraint_func=g3)

    # Constrained Optimization using Log Barrier
    # f(x
