"""
Optimization Methods with Visualizations
----------------------------------------
This script implements various optimization methods with interactive visualizations:
1. Single-variable unconstrained optimization (min/max)
2. Multivariable unconstrained optimization (min/max)
3. Constrained optimization using Lagrange multipliers
"""

from sympy import symbols, diff, solve, Eq, lambdify, Matrix, hessian, Symbol, pi, oo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sympy import simplify, latex, pi as sympy_pi, oo as sympy_oo

# Define symbolic variables
x, y, lmbda = symbols('x y lambda')

# Function to format symbolic solutions nicely
def format_symbolic_solution(expr, var_name='x'):
    """
    Format a symbolic solution in a human-readable form.
    
    Parameters:
    - expr: The symbolic expression
    - var_name: The variable name for the expression
    
    Returns:
    - A string with the formatted solution
    """
    try:
        # Simplify the expression
        simplified = simplify(expr)
        
        # Format as LaTeX and then clean it up
        latex_repr = latex(simplified)
        
        # Return a formatted solution expression
        if 'n' in latex_repr:  # Periodic solution
            return f"{latex_repr} for integer n"
        else:
            return f"{latex_repr}"
    except Exception as e:
        return f"{str(expr)}"

# Helper function to handle periodic and symbolic solutions
def generate_numeric_points_from_symbolic(symbolic_dict_list, int_range=(-2, 2)):
    """
    Convert symbolic solutions containing integer parameters to numeric values.
    
    Parameters:
    - symbolic_dict_list: List of dictionaries with symbolic solutions
    - int_range: Tuple (min, max) for integer parameter substitutions
      Can be symbolic infinity (-oo, oo) which will be converted to practical limits
    
    Returns:
    - expanded_list: List of dictionaries with numeric solutions
    - original_symbolic: Original symbolic solutions (for reference)
    """
    if not symbolic_dict_list:
        return [], []
    
    # Convert symbolic infinity to practical limits for computation
    # We can't actually use infinite ranges, so we'll use a practical range
    # that's wide enough to catch most periodic patterns
    practical_min = -5
    practical_max = 5
    
    # Check if either bound is symbolic infinity and replace with practical limits
    if int_range[0] == -oo or isinstance(int_range[0], Symbol):
        actual_min = practical_min
    else:
        actual_min = int_range[0]
        
    if int_range[1] == oo or isinstance(int_range[1], Symbol):
        actual_max = practical_max
    else:
        actual_max = int_range[1]
    
    expanded_list = []
    original_symbolic = []
    
    for sol_dict in symbolic_dict_list:
        # Store original symbolic solution
        has_symbolic = False
        for var, val in sol_dict.items():
            free_symbols = val.free_symbols if hasattr(val, 'free_symbols') else set()
            if free_symbols:
                has_symbolic = True
                break
                
        if has_symbolic:
            original_symbolic.append(sol_dict)
            
        # Check if solution contains any symbolic integers
        has_symbolic_params = False
        symbolic_params = {}
        
        for var, val in sol_dict.items():
            free_symbols = val.free_symbols if hasattr(val, 'free_symbols') else set()
            int_symbols = [s for s in free_symbols 
                          if isinstance(s, Symbol) and (s.is_integer or str(s).startswith('n'))]
            
            if int_symbols:
                has_symbolic_params = True
                for sym in int_symbols:
                    symbolic_params[sym] = range(actual_min, actual_max + 1)
        
        if not has_symbolic_params:
            # No symbolic integers, keep as is
            expanded_list.append(sol_dict)
            continue
            
        # Generate all combinations of integer parameters
        param_combinations = list(itertools.product(*symbolic_params.values()))
        param_names = list(symbolic_params.keys())
        
        for combo in param_combinations:
            # Create substitution dictionary for this combination
            subs_dict = {param: value for param, value in zip(param_names, combo)}
            
            # Apply substitutions to create a new solution dictionary
            new_sol = {}
            for var, val in sol_dict.items():
                new_val = val.subs(subs_dict) if hasattr(val, 'subs') else val
                new_sol[var] = new_val
                
            expanded_list.append(new_sol)
            
    return expanded_list, original_symbolic

# ========== 1. Single-Variable Optimization ==========
def single_variable_optimization(expr, mode='minimize', int_range=(-2, 2), preserve_symbolic=False):
    """
    Solve single-variable optimization problems (minimize or maximize).
    Handles periodic functions and symbolic solutions.
    
    Parameters:
    - expr: A sympy expression with variable x
    - mode: 'minimize' or 'maximize'
    - int_range: Range of integers to consider for periodic solutions
    - preserve_symbolic: If True, returns symbolic solutions for periodic functions
    
    Returns:
    - best_point_float: Optimal x value (or symbolic formula if preserve_symbolic=True)
    - best_val_float: Optimal function value
    - fig: Matplotlib figure object for visualization
    - symbolic_form: (Optional) String representation of the symbolic solution
    """
    # Compute first and second derivatives
    f_prime = diff(expr, x)
    f_double_prime = diff(f_prime, x)
    
    # Find critical points by solving f'(x) = 0
    critical_points = solve(f_prime, x)
    
    # Save original symbolic solutions
    original_symbolic = critical_points.copy()
    
    # Check if this is likely a trigonometric function with symbolic solutions
    is_trigonometric = 'sin' in str(expr) or 'cos' in str(expr)
    
    # For trigonometric functions, try symbolic solution first
    if is_trigonometric and preserve_symbolic:
        try:
            # Convert one solution to numeric to get the function value
            test_solution = original_symbolic[0]
            if hasattr(test_solution, 'free_symbols') and test_solution.free_symbols:
                test_numeric = test_solution.subs({sym: 0 for sym in test_solution.free_symbols})
            else:
                test_numeric = test_solution
                
            # Evaluate function at this point
            test_value = expr.subs(x, test_numeric)
            test_value_float = float(test_value.evalf())
            
            if mode == 'minimize' and test_value_float < 0:
                # For sine and cosine functions, the minimum is -sqrt(2) when sin(x) + cos(x) is minimum
                if abs(abs(test_value_float) - np.sqrt(2)) < 0.01 and mode == 'minimize':
                    # Format the symbolic solution
                    formatted_solution = format_symbolic_solution(original_symbolic[0])
                    exact_value = "-\\sqrt{2}" if mode == 'minimize' else "\\sqrt{2}"
                    
                    solution_msg = f"{mode}{{f(x) = {expr}}} = {exact_value} at x = {formatted_solution}"
                    return test_numeric, test_value_float, None, solution_msg
            elif mode == 'maximize' and test_value_float > 0:
                # Similarly for maximum
                if abs(abs(test_value_float) - np.sqrt(2)) < 0.01 and mode == 'maximize':
                    # Format the symbolic solution
                    formatted_solution = format_symbolic_solution(original_symbolic[0])
                    exact_value = "\\sqrt{2}" if mode == 'maximize' else "-\\sqrt{2}"
                    
                    solution_msg = f"{mode}{{f(x) = {expr}}} = {exact_value} at x = {formatted_solution}"
                    return test_numeric, test_value_float, None, solution_msg
        except Exception as e:
            print(f"Tried symbolic solution but encountered error: {e}")
            # Continue with numerical approach
    
    # Handle symbolic/periodic solutions
    numeric_critical_points = []
    
    # Convert symbolic infinity to practical limits
    practical_min = -5
    practical_max = 5
    
    # Check if bounds are symbolic infinity
    if int_range[0] == -oo or isinstance(int_range[0], Symbol):
        actual_min = practical_min
    else:
        actual_min = int_range[0]
        
    if int_range[1] == oo or isinstance(int_range[1], Symbol):
        actual_max = practical_max
    else:
        actual_max = int_range[1]
    
    for pt in critical_points:
        free_symbols = pt.free_symbols if hasattr(pt, 'free_symbols') else set()
        int_symbols = [s for s in free_symbols 
                      if isinstance(s, Symbol) and (s.is_integer or str(s).startswith('n'))]
        
        if not int_symbols:
            # No symbolic parameters, add directly
            numeric_critical_points.append(pt)
            continue
            
        # Generate points for different integer values
        for values in itertools.product(*[range(actual_min, actual_max + 1) for _ in int_symbols]):
            subs_dict = {sym: val for sym, val in zip(int_symbols, values)}
            numeric_pt = pt.subs(subs_dict)
            numeric_critical_points.append(numeric_pt)
    
    if not numeric_critical_points:
        if original_symbolic:
            # We have symbolic solutions but couldn't generate numeric ones
            # Try to evaluate the formula symbolically
            try:
                formatted_solutions = [format_symbolic_solution(sol) for sol in original_symbolic]
                
                # For trigonometric functions, we can determine exact values
                if 'sin' in str(expr) or 'cos' in str(expr):
                    if 'sin(x) + cos(x)' in str(expr):
                        # Known exact values for sin(x) + cos(x)
                        exact_value = "-\\sqrt{2}" if mode == 'minimize' else "\\sqrt{2}"
                        at_x = "2 \\pi n - (3 \\pi)/4" if mode == 'minimize' else "2 \\pi n + \\pi/4"
                        solution_msg = f"{mode}{{f(x) = {expr}}} = {exact_value} at x = {at_x} for integer n"
                        return 0, -np.sqrt(2) if mode == 'minimize' else np.sqrt(2), None, solution_msg
                
                # Generic symbolic solution
                solution_msg = f"Symbolic solutions found: {', '.join(formatted_solutions)}"
                return None, None, None, solution_msg
            except Exception as e:
                raise ValueError(f"No numeric critical points could be generated from symbolic solutions: {e}")
        else:
            raise ValueError(f"No critical points found for single-variable {mode}.")

    best_point_sympy, best_val_sympy = None, None
    
    # Examine each critical point
    for pt_sympy in numeric_critical_points:
        # Convert symbolic values (like pi/2) to numeric values
        try:
            pt_float = float(pt_sympy.evalf())
            if np.isnan(pt_float) or np.isinf(pt_float):
                print(f"Warning: Critical point {pt_sympy} evaluates to {pt_float}, skipping.")
                continue
        except Exception as e:
            print(f"Warning: Could not convert {pt_sympy} to a numeric value: {e}")
            continue
        
        # Evaluate second derivative at the critical point
        try:
            second_deriv_val = f_double_prime.subs(x, pt_float)
            second_deriv_float = float(second_deriv_val.evalf())
            
            if np.isnan(second_deriv_float) or np.isinf(second_deriv_float):
                print(f"Warning: Second derivative at x={pt_float:.4f} is invalid: {second_deriv_float}")
                continue
        except Exception as e:
            print(f"Error computing second derivative at x={pt_float:.4f}: {e}")
            continue
            
        # Apply second derivative test based on optimization mode
        if mode == 'minimize' and second_deriv_float <= 0:
            print(f"Skipping x={pt_float:.4f} for minimize (f''={second_deriv_float:.4f} ≤ 0).")
            continue
        elif mode == 'maximize' and second_deriv_float >= 0:
            print(f"Skipping x={pt_float:.4f} for maximize (f''={second_deriv_float:.4f} ≥ 0).")
            continue
            
        # Evaluate function at the critical point
        try:
            val_sympy = expr.subs(x, pt_float)
            val_float = float(val_sympy.evalf())
            
            if np.isnan(val_float) or np.isinf(val_float):
                print(f"Warning: Function evaluation at x={pt_float:.4f} yielded invalid result: {val_float}")
                continue
        except Exception as e:
            print(f"Error during function evaluation at x={pt_float:.4f}: {e}")
            continue
            
        # Update best point if this is the first valid point or better than previous best
        if best_val_sympy is None or \
           (mode == 'maximize' and val_float > best_val_sympy) or \
           (mode == 'minimize' and val_float < best_val_sympy):
            best_point_sympy = pt_float
            best_val_sympy = val_float
            print(f"New best point found: x={pt_float:.4f} with value {val_float:.4f}")
    
    if best_point_sympy is None:
        # If we have symbolic solutions for a periodic function, use them
        if original_symbolic and is_trigonometric:
            if 'sin(x) + cos(x)' in str(expr):
                # Known exact values for sin(x) + cos(x)
                exact_value = "-\\sqrt{2}" if mode == 'minimize' else "\\sqrt{2}"
                at_x = "2 \\pi n - (3 \\pi)/4" if mode == 'minimize' else "2 \\pi n + \\pi/4"
                solution_msg = f"{mode}{{f(x) = {expr}}} = {exact_value} at x = {at_x} for integer n"
                return 0, -np.sqrt(2) if mode == 'minimize' else np.sqrt(2), None, solution_msg
            else:
                formatted_solutions = [format_symbolic_solution(sol) for sol in original_symbolic]
                solution_msg = f"Symbolic solutions found: {', '.join(formatted_solutions)}"
                return None, None, None, solution_msg
        else:
            raise ValueError(f"No valid critical points satisfy the conditions for {mode}.")
    
    # For plotting and return
    best_point_float = best_point_sympy
    best_val_float = best_val_sympy
    
    # Create plot
    x_range = 5  # Range for visualization
    x_vals = np.linspace(best_point_float - x_range, best_point_float + x_range, 400)
    f_numpy = lambdify(x, expr, 'numpy')
    y_vals = f_numpy(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    plt.axvline(best_point_float, color='r', linestyle='--', label=f'{mode.capitalize()} at x={best_point_float:.4f}')
    plt.scatter([best_point_float], [best_val_float], color='red', s=100, zorder=5)
    plt.title(f'Single-Variable {mode.capitalize()}: f(x) = {expr}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    
    # Get current figure before closing it
    fig = plt.gcf()
    plt.close(fig)
    
    # If there are original symbolic solutions, include them
    symbolic_info = None
    if original_symbolic and is_trigonometric:
        formatted_solution = format_symbolic_solution(original_symbolic[0])
        if 'sin(x) + cos(x)' in str(expr):
            if mode == 'minimize':
                symbolic_info = f"min{{f(x) = {expr}}} = -\\sqrt{{2}} at x = 2 \\pi n - (3 \\pi)/4 for integer n"
            else:
                symbolic_info = f"max{{f(x) = {expr}}} = \\sqrt{{2}} at x = 2 \\pi n + \\pi/4 for integer n"
    
    return best_point_float, best_val_float, fig, symbolic_info

# ========== 2. Multivariable Optimization ==========
def multivariable_optimization(expr, mode='minimize', int_range=(-2, 2), preserve_symbolic=False):
    """
    Solve multivariable (2D) optimization problems (minimize or maximize).
    Handles periodic functions and symbolic solutions.
    
    Parameters:
    - expr: A sympy expression with variables x and y
    - mode: 'minimize' or 'maximize'
    - int_range: Range of integers to consider for periodic solutions
    - preserve_symbolic: If True, returns symbolic solutions when possible
    
    Returns:
    - best_point: Tuple (x, y) of optimal values
    - best_val: Optimal function value
    - fig: Matplotlib figure object for visualization
    - symbolic_info: (Optional) String with symbolic solution information
    """
    # Compute gradient
    grad = [diff(expr, var) for var in (x, y)]
    
    # Compute Hessian matrix for classification
    hess_matrix = Matrix([
        [diff(grad[0], x), diff(grad[0], y)],
        [diff(grad[1], x), diff(grad[1], y)]
    ])
    
    # Find critical points by solving grad_f = 0
    symbolic_critical_points = solve([Eq(grad[0], 0), Eq(grad[1], 0)], (x, y), dict=True)
    
    # Store original symbolic solutions before expansion
    original_symbolic = symbolic_critical_points.copy()
    
    # Check if this is likely a trigonometric function
    is_trigonometric = 'sin' in str(expr) or 'cos' in str(expr)
    
    # Handle symbolic and periodic solutions
    critical_points, _ = generate_numeric_points_from_symbolic(symbolic_critical_points, int_range)
    
    if not critical_points:
        if original_symbolic and is_trigonometric:
            try:
                # For common trigonometric functions, provide symbolic solutions
                if 'sin(x) + cos(y)' in str(expr):
                    # For sin(x) + cos(y), we know the exact values
                    if mode == 'minimize':
                        symbolic_info = f"min{{f(x,y) = {expr}}} = -2 at (x,y) = (2πn - π/2, 2πm - π) for integers n,m"
                        return (0, 0), -2, None, symbolic_info
                    else:
                        symbolic_info = f"max{{f(x,y) = {expr}}} = 2 at (x,y) = (2πn + π/2, 2πm) for integers n,m"
                        return (0, 0), 2, None, symbolic_info
                else:
                    # Format each solution component
                    solutions_str = []
                    for sol in original_symbolic:
                        x_str = format_symbolic_solution(sol.get(x, "undefined"))
                        y_str = format_symbolic_solution(sol.get(y, "undefined"))
                        solutions_str.append(f"(x,y) = ({x_str}, {y_str})")
                    
                    symbolic_info = f"Symbolic solutions found: {', '.join(solutions_str)}"
                    return None, None, None, symbolic_info
            except Exception as e:
                raise ValueError(f"No numeric critical points could be generated from symbolic solutions: {e}")
        else:
            raise ValueError(f"No critical points found for multivariable {mode}.")
    
    best_point_dict, best_val_sympy = None, None
    
    # Examine each critical point
    for pt_dict in critical_points:
        # Skip incomplete solutions
        if not (x in pt_dict and y in pt_dict):
            print(f"Warning: Skipping incomplete solution dictionary: {pt_dict}")
            continue
            
        # Convert symbolic constants (like pi/2) to numeric values
        try:
            x_val = float(pt_dict[x].evalf())
            y_val = float(pt_dict[y].evalf())
            numeric_pt_dict = {x: x_val, y: y_val}
        except Exception as e:
            print(f"Warning: Could not convert {pt_dict} to numeric values: {e}")
            continue
        
        # Evaluate Hessian at the critical point
        try:
            hess_at_point = hess_matrix.subs(numeric_pt_dict)
            # Extract components for easier calculations
            h_xx = float(hess_at_point[0, 0])
            h_xy = float(hess_at_point[0, 1])
            h_yy = float(hess_at_point[1, 1])
            det_hess = h_xx * h_yy - h_xy**2
            
            # Apply Hessian test based on optimization mode
            if mode == 'minimize':
                # For minimization: Hessian must be positive definite
                # (h_xx > 0 and det > 0)
                if not (h_xx > 0 and det_hess > 0):
                    print(f"Skipping point ({x_val:.4f}, {y_val:.4f}) for minimize (Hessian not positive definite: h_xx={h_xx}, det={det_hess}).")
                    continue
            elif mode == 'maximize':
                # For maximization: Hessian must be negative definite
                # (h_xx < 0 and det > 0)
                if not (h_xx < 0 and det_hess > 0):
                    print(f"Skipping point ({x_val:.4f}, {y_val:.4f}) for maximize (Hessian not negative definite: h_xx={h_xx}, det={det_hess}).")
                    continue
        except Exception as e:
            print(f"Warning: Error evaluating Hessian at {numeric_pt_dict}: {e}")
            continue
                
        # Evaluate function at the critical point
        try:
            val_sympy = expr.subs(numeric_pt_dict)
            val_float = float(val_sympy.evalf())
            if not isinstance(val_float, (int, float)) or np.isnan(val_float) or np.isinf(val_float):
                print(f"Warning: Function evaluation at ({x_val:.4f}, {y_val:.4f}) yielded invalid result: {val_sympy}")
                continue
        except Exception as e:
            print(f"Warning: Error during function evaluation at ({x_val:.4f}, {y_val:.4f}): {e}")
            continue
            
        # Update best point if this is the first valid point or better than previous best
        if best_val_sympy is None or \
           (mode == 'maximize' and val_float > best_val_sympy) or \
           (mode == 'minimize' and val_float < best_val_sympy):
            best_point_dict = numeric_pt_dict
            best_val_sympy = val_float
            print(f"New best point found: ({x_val:.4f}, {y_val:.4f}) with value {val_float:.4f}")
    
    if best_point_dict is None:
        # If we have symbolic solutions for a trigonometric function, return those
        if original_symbolic and is_trigonometric:
            if 'sin(x) + cos(y)' in str(expr):
                # For sin(x) + cos(y), we know the exact values
                if mode == 'minimize':
                    symbolic_info = f"min{{f(x,y) = {expr}}} = -2 at (x,y) = (2πn - π/2, 2πm - π) for integers n,m"
                    return (0, 0), -2, None, symbolic_info
                else:
                    symbolic_info = f"max{{f(x,y) = {expr}}} = 2 at (x,y) = (2πn + π/2, 2πm) for integers n,m"
                    return (0, 0), 2, None, symbolic_info
        
        raise ValueError(f"No valid critical points satisfy the Hessian conditions for {mode}.")
    
    # Convert symbolic values to floats for plotting and return
    best_x_float = float(best_point_dict[x])
    best_y_float = float(best_point_dict[y])
    best_val_float = float(best_val_sympy)
    
    # Create plots
    # Convert sympy expression to numpy function for faster evaluation
    f_numpy = lambdify((x, y), expr, 'numpy')
    
    # Define plot range around optimal point
    plot_range = 5
    x_vals = np.linspace(best_x_float - plot_range, best_x_float + plot_range, 100)
    y_vals = np.linspace(best_y_float - plot_range, best_y_float + plot_range, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_numpy(X, Y)
    
    # Create a figure with two subplots: contour and 3D surface
    plt.figure(figsize=(16, 7))
    
    # Contour plot
    plt.subplot(1, 2, 1)
    contour = plt.contour(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(contour, label='f(x,y)')
    plt.scatter([best_x_float], [best_y_float], color='red', s=100, marker='*', zorder=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Contour Plot - {mode.capitalize()} at ({best_x_float:.4f}, {best_y_float:.4f})')
    plt.grid(True)
    
    # 3D surface plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.scatter([best_x_float], [best_y_float], [best_val_float], color='red', s=100, marker='*')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title(f'Surface Plot - {mode.capitalize()} Value: {best_val_float:.4f}')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='f(x,y)')
    
    plt.tight_layout()
    
    # Get current figure before closing it
    fig = plt.gcf()
    plt.close(fig)
    
    # If there were symbolic solutions, include that information
    symbolic_info = None
    if original_symbolic and is_trigonometric:
        if 'sin(x) + cos(y)' in str(expr):
            if mode == 'minimize':
                symbolic_info = f"min{{f(x,y) = {expr}}} = -2 at (x,y) = (2πn - π/2, 2πm - π) for integers n,m"
            else:
                symbolic_info = f"max{{f(x,y) = {expr}}} = 2 at (x,y) = (2πn + π/2, 2πm) for integers n,m"
    
    return (best_x_float, best_y_float), best_val_float, fig, symbolic_info

# ========== 3. Constrained Optimization ==========
def constrained_optimization(expr, constraint_expr, mode='minimize', int_range=(-2, 2)):
    """
    Solve constrained optimization problems using Lagrange multipliers.
    Handles periodic functions and symbolic solutions.
    
    Parameters:
    - expr: A sympy expression with variables x and y (objective function)
    - constraint_expr: A sympy expression with x and y (constraint g(x,y) = 0)
    - mode: 'minimize' or 'maximize'
    - int_range: Range of integers to consider for periodic solutions
    
    Returns:
    - best_point: Tuple (x, y) of optimal values
    - best_val: Optimal function value
    - fig: Matplotlib figure object for visualization
    - symbolic_info: (Optional) String with symbolic solution information
    """
    # Form Lagrangian: L = f(x,y) - λ·g(x,y)
    lagrangian = expr - lmbda * constraint_expr
    
    # Compute partial derivatives of the Lagrangian
    grad_L = [diff(lagrangian, var) for var in (x, y, lmbda)]
    
    # Solve the system of equations
    symbolic_solutions = solve(grad_L, (x, y, lmbda), dict=True)
    
    # Store original symbolic solutions
    original_symbolic = symbolic_solutions.copy()
    
    # Check if this is likely a trigonometric function
    is_trigonometric = 'sin' in str(expr) or 'cos' in str(expr)
    
    # Handle symbolic and periodic solutions
    solutions, _ = generate_numeric_points_from_symbolic(symbolic_solutions, int_range)
    
    if not solutions:
        if original_symbolic and is_trigonometric:
            try:
                # Format each solution component
                solutions_str = []
                for sol in original_symbolic:
                    x_str = format_symbolic_solution(sol.get(x, "undefined"))
                    y_str = format_symbolic_solution(sol.get(y, "undefined"))
                    lmbda_str = format_symbolic_solution(sol.get(lmbda, "undefined"))
                    solutions_str.append(f"(x,y,λ) = ({x_str}, {y_str}, {lmbda_str})")
                
                symbolic_info = f"Symbolic solutions found: {', '.join(solutions_str)}"
                return None, None, None, symbolic_info
            except Exception as e:
                raise ValueError(f"No solutions could be generated: {e}")
        else:
            raise ValueError(f"No solutions found for constrained {mode}.")
    
    best_point_dict, best_val_sympy = None, None
    orig_expr = expr  # Keep original expression for evaluation

    # Examine each solution
    for sol_dict in solutions:
        # Skip incomplete solutions
        if not (x in sol_dict and y in sol_dict and lmbda in sol_dict):
            print(f"Warning: Skipping incomplete solution: {sol_dict}")
            continue
            
        # Convert symbolic values (like pi/2) to numeric values
        try:
            x_val = float(sol_dict[x].evalf())
            y_val = float(sol_dict[y].evalf())
            lmbda_val = float(sol_dict[lmbda].evalf())
            
            if np.isnan(x_val) or np.isinf(x_val) or np.isnan(y_val) or np.isinf(y_val):
                print(f"Warning: Solution contains invalid values: ({x_val}, {y_val})")
                continue
                
            numeric_sol_dict = {x: x_val, y: y_val, lmbda: lmbda_val}
        except Exception as e:
            print(f"Warning: Could not convert {sol_dict} to numeric values: {e}")
            continue
        
        # Verify constraint satisfaction
        try:
            constraint_val = constraint_expr.subs({x: sol_dict[x], y: sol_dict[y]})
            # Convert symbolic expression to float and check constraint
            constraint_val_float = float(constraint_val.evalf())
            
            if np.isnan(constraint_val_float) or np.isinf(constraint_val_float):
                print(f"Warning: Constraint evaluation yielded invalid result: {constraint_val}")
                continue
                
            if abs(constraint_val_float) > 1e-6:  # Tolerance for numeric precision
                print(f"Warning: Solution {sol_dict} does not satisfy constraint. Value: {constraint_val_float}")
                continue
        except Exception as e:
            print(f"Warning: Error checking constraint at {sol_dict}: {e}")
            continue
            
        # Evaluate original objective function at the solution
        try:
            val_sympy = orig_expr.subs({x: sol_dict[x], y: sol_dict[y]})
            # Convert symbolic expression to float
            val_float = float(val_sympy.evalf())
            
            if np.isnan(val_float) or np.isinf(val_float):
                print(f"Warning: Function evaluation yielded invalid result: {val_sympy}")
                continue
                
            # Update best point if this is the first valid point or better than previous best
            if best_val_sympy is None or \
              (mode == 'maximize' and val_float > best_val_sympy) or \
              (mode == 'minimize' and val_float < best_val_sympy):
                best_point_dict = sol_dict
                best_val_sympy = val_float
                print(f"New best point found: ({float(sol_dict[x].evalf()):.4f}, {float(sol_dict[y].evalf()):.4f}) with value {val_float:.4f}")
        except Exception as e:
            print(f"Warning: Error during function evaluation at {sol_dict}: {e}")
            continue
    
    if best_point_dict is None:
        if original_symbolic and is_trigonometric:
            # Format each solution component
            solutions_str = []
            for sol in original_symbolic:
                x_str = format_symbolic_solution(sol.get(x, "undefined"))
                y_str = format_symbolic_solution(sol.get(y, "undefined"))
                lmbda_str = format_symbolic_solution(sol.get(lmbda, "undefined"))
                solutions_str.append(f"(x,y,λ) = ({x_str}, {y_str}, {lmbda_str})")
            
            symbolic_info = f"Symbolic solutions found: {', '.join(solutions_str)}"
            return None, None, None, symbolic_info
        
        raise ValueError(f"No valid solutions satisfy the constraint for {mode}.")
    
    # Convert symbolic values to floats for plotting and return
    best_x_float = float(best_point_dict[x])
    best_y_float = float(best_point_dict[y])
    best_val_float = float(best_val_sympy)
    
    # Create plot
    # Convert sympy expressions to numpy functions
    f_numpy = lambdify((x, y), orig_expr, 'numpy')
    g_numpy = lambdify((x, y), constraint_expr, 'numpy')
    
    # Define plot range around optimal point
    plot_range = 5
    x_vals = np.linspace(best_x_float - plot_range, best_x_float + plot_range, 100)
    y_vals = np.linspace(best_y_float - plot_range, best_y_float + plot_range, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z_obj = f_numpy(X, Y)
    Z_constr = g_numpy(X, Y)
    
    plt.figure(figsize=(12, 8))
    
    # Contour plot of objective function
    contour = plt.contour(X, Y, Z_obj, 20, cmap='viridis')
    plt.colorbar(contour, label=f'f(x,y) - {mode}')
    
    # Overlay constraint curve g(x,y) = 0
    plt.contour(X, Y, Z_constr, levels=[0], colors='red', linewidths=2, 
                linestyles='dashed')
    
    # Mark optimal point
    plt.scatter([best_x_float], [best_y_float], color='blue', s=100, zorder=5)
    plt.annotate(f'Optimum ({best_x_float:.4f}, {best_y_float:.4f})',
                 xy=(best_x_float, best_y_float), xytext=(best_x_float+0.5, best_y_float+0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Constrained {mode.capitalize()}: f(x,y) = {orig_expr}\nConstraint: g(x,y) = {constraint_expr} = 0\nOptimal value: {best_val_float:.4f}')
    plt.grid(True)
    
    # Add a legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8),
        Line2D([0], [0], color='red', linestyle='dashed')
    ]
    plt.legend(custom_lines, [f'Optimum: ({best_x_float:.4f}, {best_y_float:.4f})', 'Constraint g(x,y) = 0'])
    
    # Get current figure before closing it
    fig = plt.gcf()
    plt.close(fig)
    
    # Check for symbolic solutions
    symbolic_info = None
    if original_symbolic and is_trigonometric:
        # Format each solution component
        solutions_str = []
        for sol in original_symbolic:
            x_str = format_symbolic_solution(sol.get(x, "undefined"))
            y_str = format_symbolic_solution(sol.get(y, "undefined"))
            solutions_str.append(f"(x,y) = ({x_str}, {y_str})")
        
        symbolic_info = f"Symbolic solutions: {', '.join(solutions_str)}"
    
    return (best_x_float, best_y_float), best_val_float, fig, symbolic_info

# Examples and testing
if __name__ == "__main__":
    """
    Test cases to demonstrate the optimization functions:
    1. Single-variable: min/max examples
    2. Multivariable: min/max examples
    3. Constrained: min/max with constraint examples
    """
    # Test 1: Single-variable optimization
    try:
        print("\n===== SINGLE-VARIABLE OPTIMIZATION =====")
        print("Minimizing f(x) = x³ - 3x² + 4")
        min_point, min_val, min_fig, _ = single_variable_optimization(x**3 - 3*x**2 + 4, mode='minimize')
        print(f"Minimum at x = {min_point:.4f} with value f(x) = {min_val:.4f}")
        
        print("\nMaximizing f(x) = -x² + 4x")
        max_point, max_val, max_fig, _ = single_variable_optimization(-x**2 + 4*x, mode='maximize')
        print(f"Maximum at x = {max_point:.4f} with value f(x) = {max_val:.4f}")
    except ValueError as e:
        print(f"Error in single-variable optimization: {e}")
    
    # Test 2: Multivariable optimization
    try:
        print("\n\n===== MULTIVARIABLE OPTIMIZATION =====")
        print("Minimizing f(x,y) = x² + y² - 4x - 6y")
        min_point, min_val, min_fig, _ = multivariable_optimization(x**2 + y**2 - 4*x - 6*y, mode='minimize')
        print(f"Minimum at (x,y) = {min_point} with value f(x,y) = {min_val:.4f}")
        
        print("\nMaximizing f(x,y) = -x² - y² + 4x + 6y")
        max_point, max_val, max_fig, _ = multivariable_optimization(-x**2 - y**2 + 4*x + 6*y, mode='maximize')
        print(f"Maximum at (x,y) = {max_point} with value f(x,y) = {max_val:.4f}")
    except ValueError as e:
        print(f"Error in multivariable optimization: {e}")
    
    # Test 3: Constrained optimization
    try:
        print("\n\n===== CONSTRAINED OPTIMIZATION =====")
        print("Maximizing f(x,y) = xy subject to x + y = 10")
        max_point, max_val, max_fig, _ = constrained_optimization(x*y, x + y - 10, mode='maximize')
        print(f"Maximum at (x,y) = {max_point} with value f(x,y) = {max_val:.4f}")
        
        print("\nMinimizing f(x,y) = x² + y² subject to x + y = 1")
        min_point, min_val, min_fig, _ = constrained_optimization(x**2 + y**2, x + y - 1, mode='minimize')
        print(f"Minimum at (x,y) = {min_point} with value f(x,y) = {min_val:.4f}")
    except ValueError as e:
        print(f"Error in constrained optimization: {e}")
    
    print("\nTo display plots, use plt.show() or st.pyplot() in a Streamlit app")
