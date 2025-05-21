import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib import cm
import sympy as sp
from sympy import symbols, lambdify, diff, hessian, Matrix, solve, pi

# ----------- Helper plotting functions -----------

def plot_2d_function(f, x_range, x_min=None, mode='minimize', return_fig=False):
    """
    Create a 2D plot of a single-variable function with optimization point marked.
    
    Parameters:
    - f: Function to plot
    - x_range: Tuple (min, max) for x-axis
    - x_min: Optimal point (if known)
    - mode: 'minimize' or 'maximize'
    - return_fig: If True, returns the figure object instead of showing it
    
    Returns:
    - fig: Matplotlib figure object if return_fig=True
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(*x_range, 400)
    y = f(x)
    
    # Plot the function
    ax.plot(x, y, label="f(x)", linewidth=2.5)
      # Mark the optimal point if provided
    if x_min is not None:
        try:
            # Ensure x_min is a scalar
            if isinstance(x_min, (list, np.ndarray)):
                x_min = x_min[0]
            
            f_val = f(x_min)
            
            point_label = "Minimum" if mode == 'minimize' else "Maximum"
            ax.plot(x_min, f_val, 'ro', markersize=8, label=f"{point_label}: x={x_min:.4f}")
            
            # Add vertical line from x-axis to the optimal point
            ax.vlines(x=x_min, ymin=min(y), ymax=f_val, linestyles='dashed', colors='red', alpha=0.7)
            
            # Add annotation with function value
            # Convert f_val to float explicitly to avoid formatting issues with numpy arrays
            f_val_float = float(f_val) if isinstance(f_val, np.ndarray) else f_val
            ax.annotate(f'f({x_min:.4f}) = {f_val_float:.4f}', 
                     xy=(x_min, f_val),
                     xytext=(10, -30), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))
        except Exception as e:
            print(f"Warning: Could not plot optimal point: {e}")
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"{'Minimization' if mode=='minimize' else 'Maximization'} of f(x)", fontsize=14)
    plt.tight_layout()
    
    if return_fig:
        return fig
    plt.show()
    return None

def plot_contour_and_surface(f, bounds, optimum=None, constraint_func=None, mode='minimize', return_fig=False):
    """
    Create contour and surface plots for multivariable optimization.
    
    Parameters:
    - f: Function to plot (takes vector input)
    - bounds: List of tuples [(x_min, x_max), (y_min, y_max)]
    - optimum: Optimal point [x, y]
    - constraint_func: Constraint function g(x,y) ≥ 0 (if applicable)
    - mode: 'minimize' or 'maximize'
    - return_fig: If True, returns the figure object instead of showing it
    
    Returns:
    - fig: Matplotlib figure object if return_fig=True
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Generate grid for visualization
    x = np.linspace(*bounds[0], 100)
    y = np.linspace(*bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Handle different types of function inputs
    try:
        if callable(f):
            # Handle 2D array input: f([X, Y])
            try:
                Z = f(np.array([X, Y]))
            except:
                # Handle element-wise evaluation: f(point) for each point
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
        else:
            # Assume f is a lambda that takes individual x,y inputs
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = f(X[i, j], Y[i, j])
    except Exception as e:
        print(f"Warning: Error generating Z values: {e}")
        # Fall back to element-wise evaluation if other methods fail
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
                except:
                    Z[i, j] = np.nan

    # Contour Plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(X, Y, Z, levels=30, cmap='viridis')
    fig.colorbar(contour, ax=ax1, shrink=0.8)
    
    # Plot constraint if provided
    if constraint_func is not None:
        try:
            # Try different approaches to evaluate constraint
            try:
                C = constraint_func(np.array([X, Y]))
            except:
                C = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        C[i, j] = constraint_func(np.array([X[i, j], Y[i, j]]))
                        
            # Plot the constraint g(x,y) = 0
            ax1.contour(X, Y, C, levels=[0], colors='red', linewidths=2)
            
            # Add a red patch for the legend
            constraint_patch = mpatches.Patch(color='red', label='Constraint g(x,y) = 0')
            handles, labels = ax1.get_legend_handles_labels()
            handles.append(constraint_patch)
            labels.append('Constraint g(x,y) = 0')
            ax1.legend(handles, labels, loc='upper right')
        except Exception as e:
            print(f"Warning: Could not plot constraint: {e}")
    
    # Mark optimum point if provided
    if optimum is not None:
        try:
            point_label = "Minimum" if mode == 'minimize' else "Maximum"
            ax1.plot(optimum[0], optimum[1], 'ro', markersize=8, 
                   label=f"{point_label}: ({optimum[0]:.4f}, {optimum[1]:.4f})")
            ax1.annotate(f'f = {f(np.array(optimum)):.4f}', 
                       xy=(optimum[0], optimum[1]), 
                       xytext=(10, 10),
                       textcoords='offset points', 
                       arrowprops=dict(arrowstyle='->'))
            
            # Create a legend
            ax1.legend(loc='best')
        except Exception as e:
            print(f"Warning: Could not plot optimal point on contour: {e}")
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title("Contour Plot")
    ax1.grid(True, alpha=0.3)
    
    # Surface Plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot the function surface
    try:
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True, rstride=2, cstride=2)
        fig.colorbar(surf, ax=ax2, shrink=0.6)
    
        # Mark optimum point on surface if provided
        if optimum is not None:
            try:
                z_val = f(np.array(optimum))
                ax2.scatter(optimum[0], optimum[1], z_val, color='red', s=50, label='Optimum')
            except Exception as e:
                print(f"Warning: Could not plot optimal point on surface: {e}")
    except Exception as e:
        print(f"Warning: Could not create surface plot: {e}")
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_title("3D Surface Plot")
    
    plt.tight_layout()
    fig.suptitle(f"{'Minimization' if mode=='minimize' else 'Maximization'} of f(x,y)", 
                fontsize=14, y=1.05)
    
    if return_fig:
        return fig
    plt.show()
    return None

def plot_optimization_trajectory(f, path, bounds, constraint_func=None, mode='minimize', return_fig=False):
    """
    Visualize the optimization trajectory on a contour plot.
    
    Parameters:
    - f: Function being optimized
    - path: List of points visited during optimization
    - bounds: List of tuples [(x_min, x_max), (y_min, y_max)]
    - constraint_func: Constraint function g(x,y) ≥ 0 (if applicable)
    - mode: 'minimize' or 'maximize'
    - return_fig: Whether to return the figure instead of displaying it
    
    Returns:
    - fig: Matplotlib figure if return_fig=True
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate grid for visualization
    x = np.linspace(*bounds[0], 100)
    y = np.linspace(*bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values
    try:
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    except:
        # Try alternate approach
        try:
            Z = f(np.array([X, Y]))
        except Exception as e:
            print(f"Warning: Error generating Z values: {e}")
            Z = np.zeros_like(X)
    
    # Create contour plot
    contour = ax.contour(X, Y, Z, levels=30, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    
    # Plot constraint if provided
    if constraint_func is not None:
        try:
            C = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    C[i, j] = constraint_func(np.array([X[i, j], Y[i, j]]))
            
            ax.contour(X, Y, C, levels=[0], colors='red', linewidths=2)
            constraint_patch = mpatches.Patch(color='red', label='Constraint g(x,y) = 0')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(constraint_patch)
            labels.append('Constraint g(x,y) = 0')
        except Exception as e:
            print(f"Warning: Could not plot constraint: {e}")
    
    # Plot optimization path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'o-', color='blue', linewidth=1.5, markersize=6, label='Optimization path')
    
    # Mark start and end points
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Starting point')
    ax.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, label='Final point')
    
    # Add annotations
    for i, point in enumerate(path):
        ax.annotate(f"{i}", (point[0], point[1]), fontsize=8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Optimization Trajectory ({mode})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    plt.show()
    return None

# ----------- Numerical Methods -----------

# Steepest Descent (Gradient Descent)
def steepest_descent(f, grad_f, x0, lr=0.1, max_iter=100, tol=1e-6, path_history=False):
    """
    Optimize function using steepest descent (gradient descent).
    
    Parameters:
    - f: Function to optimize
    - grad_f: Gradient function
    - x0: Initial point
    - lr: Learning rate
    - max_iter: Maximum iterations
    - tol: Convergence tolerance
    - path_history: Whether to return the optimization path history
    
    Returns:
    - x: Optimal point
    - path: List of points visited during optimization (if path_history=True)
    """
    x = x0.copy()
    path = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        # Update rule: x = x - lr * grad
        x_new = x - lr * grad
        
        # Check if we're making progress
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        
        if path_history:
            path.append(x.copy())
    
    if path_history:
        return x, path
    return x

# Newton's Method
def newtons_method(f, grad_f, hess_f, x0, max_iter=10, tol=1e-6, path_history=False):
    """
    Optimize function using Newton's method.
    
    Parameters:
    - f: Function to optimize
    - grad_f: Gradient function
    - hess_f: Hessian function
    - x0: Initial point
    - max_iter: Maximum iterations
    - tol: Convergence tolerance
    - path_history: Whether to return the optimization path history
    
    Returns:
    - x: Optimal point
    - path: List of points visited during optimization (if path_history=True)
    """
    x = x0.copy()
    path = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        hess = hess_f(x)
        
        # Add regularization if Hessian is ill-conditioned
        min_eig = np.min(np.linalg.eigvals(hess))
        if min_eig < 1e-6:
            # Add small diagonal term to make positive definite
            hess = hess + (1e-6 - min_eig) * np.eye(len(x))
        
        try:
            # Newton's direction: p = -H⁻¹∇f
            p = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            # Fallback if linear system can't be solved
            p = -grad
        
        # Line search (simple backtracking)
        alpha = 1.0
        while alpha > 1e-10:
            x_new = x + alpha * p
            if f(x_new) < f(x):
                break
            alpha *= 0.5
        
        # If line search failed, use regular gradient descent step
        if alpha <= 1e-10:
            x_new = x - 0.1 * grad
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        
        if path_history:
            path.append(x.copy())
    
    if path_history:
        return x, path
    return x

# Penalty Method (Exterior)
def exterior_penalty_method(f, constraint, x0, r=10, max_iter=5, path_history=False):
    """
    Optimize constrained function using exterior penalty method.
    
    Parameters:
    - f: Objective function
    - constraint: Constraint function g(x) ≥ 0
    - x0: Initial point
    - r: Penalty parameter
    - max_iter: Maximum iterations
    - path_history: Whether to return the optimization path history
    
    Returns:
    - x: Optimal point
    - path: List of points visited during optimization (if path_history=True)
    """
    x = x0.copy()
    path = [x.copy()]
    
    for i in range(max_iter):
        def penalized_function(x):
            return f(x) + r * max(0, constraint(x))**2

        result = minimize(penalized_function, x)
        x = result.x
        r *= 10  # Increase penalty parameter
        
        if path_history:
            path.append(x.copy())
    
    if path_history:
        return x, path
    return x

# Log Barrier Method (Interior)
def log_barrier_method(f, constraint, x0, t=1, mu=10, max_iter=10, path_history=False):
    """
    Optimize constrained function using log barrier method.
    
    Parameters:
    - f: Objective function
    - constraint: Constraint function g(x) ≤ 0 (note this is opposite to penalty method)
    - x0: Initial point (must be in the feasible region)
    - t: Barrier parameter
    - mu: Parameter growth rate
    - max_iter: Maximum iterations
    - path_history: Whether to return the optimization path history
    
    Returns:
    - x: Optimal point
    - path: List of points visited during optimization (if path_history=True)
    """
    x = x0.copy()
    path = [x.copy()]
    
    for i in range(max_iter):
        def barrier_obj(x):
            # Ensure we're in the feasible region (-constraint(x) > 0)
            constr_val = -constraint(x)
            if constr_val <= 0:
                return float('inf')  # Return infinity for infeasible points
            return t * f(x) - np.log(constr_val)

        try:
            res = minimize(barrier_obj, x, method='BFGS')
            if res.success:
                x = res.x
                t *= mu  # Increase barrier parameter
            else:
                # If optimization failed, try different method
                res = minimize(barrier_obj, x, method='Nelder-Mead')
                if res.success:
                    x = res.x
                    t *= mu
        except Exception as e:
            print(f"Warning: Optimization step failed: {e}")
            break
            
        if path_history:
            path.append(x.copy())
    
    if path_history:
        return x, path
    return x

# Automatic numerical gradient and Hessian computation
def numerical_gradient(f, x, h=1e-8):
    """
    Compute numerical gradient of function f at point x.
    
    Parameters:
    - f: Function to differentiate
    - x: Point to evaluate gradient at
    - h: Step size for finite difference
    
    Returns:
    - grad: Numerical gradient vector
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad

def numerical_hessian(f, x, h=1e-4):
    """
    Compute numerical Hessian of function f at point x.
    
    Parameters:
    - f: Function to differentiate
    - x: Point to evaluate Hessian at
    - h: Step size for finite difference
    
    Returns:
    - hess: Numerical Hessian matrix
    """
    n = len(x)
    hess = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # For diagonal elements, use standard second derivative formula
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                hess[i, j] = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h ** 2)
            else:
                # For off-diagonal elements, use mixed partial derivative formula
                x_plus_plus = x.copy()
                x_plus_minus = x.copy()
                x_minus_plus = x.copy()
                x_minus_minus = x.copy()
                
                x_plus_plus[i] += h
                x_plus_plus[j] += h
                
                x_plus_minus[i] += h
                x_plus_minus[j] -= h
                
                x_minus_plus[i] -= h
                x_minus_plus[j] += h
                
                x_minus_minus[i] -= h
                x_minus_minus[j] -= h
                
                hess[i, j] = (f(x_plus_plus) - f(x_plus_minus) - f(x_minus_plus) + f(x_minus_minus)) / (4 * h ** 2)
    
    return hess

# Helper functions for symbolic differentiation
def create_symbolic_functions(expr_str, var_symbols):
    """
    Create symbolic function, gradient, and Hessian from expression string.
    
    Parameters:
    - expr_str: String representation of the function
    - var_symbols: List of sympy symbols
    
    Returns:
    - f: Numerical function
    - grad_f: Numerical gradient function
    - hess_f: Numerical Hessian function
    """
    try:
        expr = sp.sympify(expr_str)
        
        # Create gradient expressions
        grad_expr = [sp.diff(expr, var) for var in var_symbols]
        
        # Create Hessian expression
        hess_expr = sp.hessian(expr, var_symbols)
        
        # Convert to numerical functions
        f = sp.lambdify(var_symbols, expr, 'numpy')
        grad_f = sp.lambdify(var_symbols, grad_expr, 'numpy')
        hess_f = sp.lambdify(var_symbols, hess_expr, 'numpy')        # Wrap functions to handle array inputs
        def f_wrapper(x):
            if isinstance(x, np.ndarray):
                if x.ndim == 1:
                    # Handle both single variable and multivariable cases
                    if len(var_symbols) == 1:
                        # For single variable function with array input (vectorized)
                        if len(x) > 1:  # If it's an array of multiple values
                            return np.array([f(float(xi)) for xi in x])
                        else:  # Single element array
                            return f(float(x[0]))
                    else:
                        # For multivariable function
                        return f(*x)
                elif x.ndim == 2 and len(var_symbols) == 1:
                    # Handle 2D arrays for single variable plots (like meshgrid output)
                    return np.array([f(float(xi)) for xi in x.flatten()]).reshape(x.shape)
                else:
                    # For 2D arrays with multiple variables
                    return f(x[0], x[1])  # For 2D arrays like meshgrid
            return f(x)
        
        def grad_wrapper(x):
            if isinstance(x, np.ndarray) and x.ndim == 1:
                return np.array(grad_f(*x))
            return np.array(grad_f(x))
        
        def hess_wrapper(x):
            if isinstance(x, np.ndarray) and x.ndim == 1:
                return np.array(hess_f(*x))
            return np.array(hess_f(x))
        
        return f_wrapper, grad_wrapper, hess_wrapper
    
    except Exception as e:
        print(f"Error creating symbolic functions: {e}")
        # Return fallback numerical functions
        def f_fallback(x):
            # Implementation depends on the expected input format
            return eval(expr_str, globals(), {"x": x[0], "y": x[1] if len(x) > 1 else 0})
        
        return f_fallback, lambda x: numerical_gradient(f_fallback, x), lambda x: numerical_hessian(f_fallback, x)


# ----------- Interface Functions for Streamlit -----------

def optimize_single_variable(expr_str, mode='minimize', x0=None, method='newton'):
    """
    Optimize a single variable function using specified method.
    
    Parameters:
    - expr_str: String representation of the function (in terms of x)
    - mode: 'minimize' or 'maximize'
    - x0: Initial guess (if None, defaults are used)
    - method: 'gradient' or 'newton'
    
    Returns:
    - x_opt: Optimal x value
    - f_opt: Optimal function value
    - fig: Figure object for plotting
    - history: Optimization path history (if available)
    """
    x_sym = sp.symbols('x')
    
    # Create functions
    try:
        f, grad_f, hess_f = create_symbolic_functions(expr_str, [x_sym])
    except Exception as e:
        raise ValueError(f"Could not create functions from expression: {e}")
    
    # For maximization, negate the function
    if mode == 'maximize':
        orig_f = f
        f = lambda x: -orig_f(x)
        orig_grad_f = grad_f
        grad_f = lambda x: -orig_grad_f(x)
        orig_hess_f = hess_f
        hess_f = lambda x: -orig_hess_f(x)
    
    # Default initialization
    if x0 is None:
        x0 = np.array([0.0])
    elif isinstance(x0, (int, float)):
        x0 = np.array([x0])
    
    # Perform optimization
    path = None
    if method == 'newton':
        try:
            x_opt, path = newtons_method(f, grad_f, hess_f, x0, max_iter=50, path_history=True)
        except Exception as e:
            print(f"Newton's method failed: {e}. Falling back to gradient descent.")
            x_opt, path = steepest_descent(f, grad_f, x0, lr=0.1, max_iter=100, path_history=True)
    else:  # gradient descent
        x_opt, path = steepest_descent(f, grad_f, x0, lr=0.1, max_iter=100, path_history=True)
      # Get function value (using original function for maximization)
    if mode == 'maximize':
        f_opt = float(orig_f(x_opt))
    else:
        f_opt = float(f(x_opt))
    
    # Determine appropriate plotting range
    if path:
        path_x = np.array([p[0] for p in path])
        x_range = (min(path_x) - 2, max(path_x) + 2)
    else:
        x_range = (float(x_opt[0]) - 5, float(x_opt[0]) + 5)
    
    # For plotting, always use the original function
    plot_f = orig_f if mode == 'maximize' else f
    fig = plot_2d_function(plot_f, x_range, x_opt, mode=mode, return_fig=True)
    
    # Return scalar value for x_opt instead of array for cleaner output
    return float(x_opt[0]), f_opt, fig, path

def optimize_multivariable(expr_str, mode='minimize', x0=None, method='newton'):
    """
    Optimize a multivariable function using specified method.
    
    Parameters:
    - expr_str: String representation of the function (in terms of x,y)
    - mode: 'minimize' or 'maximize'
    - x0: Initial guess (if None, defaults are used)
    - method: 'gradient', 'newton', or 'scipy'
    
    Returns:
    - x_opt: Optimal [x,y] values
    - f_opt: Optimal function value
    - fig: Figure object for plotting
    - history: Optimization path history (if available)
    """
    x_sym, y_sym = sp.symbols('x y')
    
    # Create functions
    try:
        f, grad_f, hess_f = create_symbolic_functions(expr_str, [x_sym, y_sym])
    except Exception as e:
        raise ValueError(f"Could not create functions from expression: {e}")
    
    # For maximization, negate the function
    if mode == 'maximize':
        orig_f = f
        f = lambda x: -orig_f(x)
        orig_grad_f = grad_f
        grad_f = lambda x: -orig_grad_f(x)
        orig_hess_f = hess_f
        hess_f = lambda x: -orig_hess_f(x)
    
    # Default initialization
    if x0 is None:
        x0 = np.array([0.0, 0.0])
    
    # Perform optimization
    path = None
    if method == 'newton':
        try:
            x_opt, path = newtons_method(f, grad_f, hess_f, x0, max_iter=50, path_history=True)
        except Exception as e:
            print(f"Newton's method failed: {e}. Falling back to gradient descent.")
            x_opt, path = steepest_descent(f, grad_f, x0, lr=0.1, max_iter=100, path_history=True)
    elif method == 'scipy':
        result = minimize(f, x0, method='BFGS')
        x_opt = result.x
        path = None  # SciPy doesn't provide path history by default
    else:  # gradient descent
        x_opt, path = steepest_descent(f, grad_f, x0, lr=0.1, max_iter=100, path_history=True)
    
    # Get function value (using original function for maximization)
    if mode == 'maximize':
        f_opt = orig_f(x_opt)
    else:
        f_opt = f(x_opt)
    
    # Determine appropriate plotting range
    if path:
        path_x = np.array([p[0] for p in path])
        path_y = np.array([p[1] for p in path])
        x_range = (min(path_x) - 2, max(path_x) + 2)
        y_range = (min(path_y) - 2, max(path_y) + 2)
    else:
        x_range = (x_opt[0] - 5, x_opt[0] + 5)
        y_range = (x_opt[1] - 5, x_opt[1] + 5)
    
    # For plotting, always use the original function
    plot_f = orig_f if mode == 'maximize' else f
    
    # Create contour and surface plots
    fig = plot_contour_and_surface(plot_f, [x_range, y_range], x_opt, mode=mode, return_fig=True)
    
    # Create trajectory plot if path is available
    if path:
        traj_fig = plot_optimization_trajectory(plot_f, path, [x_range, y_range], mode=mode, return_fig=True)
        return x_opt, f_opt, fig, path, traj_fig
    
    return x_opt, f_opt, fig, path

def optimize_constrained(obj_expr_str, constraint_expr_str, mode='minimize', x0=None, method='penalty'):
    """
    Optimize a constrained multivariable function.
    
    Parameters:
    - obj_expr_str: String representation of objective function (in terms of x,y)
    - constraint_expr_str: String representation of constraint function g(x,y) = 0
    - mode: 'minimize' or 'maximize'
    - x0: Initial guess (if None, defaults are used)
    - method: 'penalty' or 'barrier'
    
    Returns:
    - x_opt: Optimal [x,y] values
    - f_opt: Optimal function value
    - fig: Figure object for plotting
    - history: Optimization path history (if available)
    """
    x_sym, y_sym = sp.symbols('x y')
    
    # Create objective function
    try:
        obj_f, _, _ = create_symbolic_functions(obj_expr_str, [x_sym, y_sym])
    except Exception as e:
        raise ValueError(f"Could not create objective function: {e}")
    
    # Create constraint function g(x,y) = 0
    try:
        # For penalty/barrier methods, we need g(x) ≥ 0 or g(x) ≤ 0
        constraint_f, _, _ = create_symbolic_functions(constraint_expr_str, [x_sym, y_sym])
    except Exception as e:
        raise ValueError(f"Could not create constraint function: {e}")
    
    # For maximization, negate the objective function
    if mode == 'maximize':
        orig_obj_f = obj_f
        obj_f = lambda x: -orig_obj_f(x)
    
    # Default initialization
    if x0 is None:
        x0 = np.array([1.0, 1.0])  # Starting closer to feasible region
    
    # Perform optimization
    path = None
    if method == 'penalty':
        # For penalty method, constraint should be g(x) ≥ 0
        try:
            x_opt, path = exterior_penalty_method(obj_f, constraint_f, x0, r=10, max_iter=10, path_history=True)
        except Exception as e:
            raise ValueError(f"Penalty method failed: {e}")
    else:  # barrier method
        # For barrier method, constraint should be g(x) ≤ 0 (opposite of penalty)
        try:
            neg_constraint_f = lambda x: -constraint_f(x)
            x_opt, path = log_barrier_method(obj_f, neg_constraint_f, x0, t=1, mu=10, max_iter=10, path_history=True)
        except Exception as e:
            raise ValueError(f"Barrier method failed: {e}. Make sure your initial point is in the feasible region.")
    
    # Get function value (using original function for maximization)
    if mode == 'maximize':
        f_opt = orig_obj_f(x_opt)
    else:
        f_opt = obj_f(x_opt)
    
    # Determine appropriate plotting range
    if path:
        path_x = np.array([p[0] for p in path])
        path_y = np.array([p[1] for p in path])
        x_range = (min(path_x) - 2, max(path_x) + 2)
        y_range = (min(path_y) - 2, max(path_y) + 2)
    else:
        x_range = (x_opt[0] - 5, x_opt[0] + 5)
        y_range = (x_opt[1] - 5, x_opt[1] + 5)
    
    # For plotting, always use the original function
    plot_f = orig_obj_f if mode == 'maximize' else obj_f
    
    # Create contour and surface plots
    fig = plot_contour_and_surface(plot_f, [x_range, y_range], x_opt, constraint_func=constraint_f, mode=mode, return_fig=True)
    
    # Create trajectory plot if path is available
    if path:
        traj_fig = plot_optimization_trajectory(plot_f, path, [x_range, y_range], constraint_func=constraint_f, mode=mode, return_fig=True)
        return x_opt, f_opt, fig, path, traj_fig
    
    return x_opt, f_opt, fig, path

# ----------- Example Usages -----------

if __name__ == "__main__":
    # Single-variable example: f(x) = (x - 2)^2
    print("Single-variable optimization example:")
    x_min, f_min, _, _ = optimize_single_variable("(x - 2)**2", mode='minimize', method='newton')
    print(f"Minimum at x = {x_min:.4f} with value {f_min:.4f}")
    
    # Multivariable example: f(x, y) = (x - 1)^2 + (y + 2)^2
    print("Multivariable optimization example:")
    x_min, f_min, _, _, _ = optimize_multivariable("(x - 1)**2 + (y + 2)**2", mode='minimize', method='newton')
    print(f"Minimum at (x, y) = ({x_min[0]:.4f}, {x_min[1]:.4f}) with value {f_min:.4f}")
    
    # Constrained optimization example: maximize xy subject to x + y = 10
    print("Constrained optimization example:")
    x_min, f_min, _, _ = optimize_constrained("x*y", "x + y - 10", mode='maximize', method='penalty')
    print(f"Maximum at (x, y) = ({x_min[0]:.4f}, {x_min[1]:.4f}) with value {f_min:.4f}")
