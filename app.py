import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, SympifyError, pi, sqrt, oo
from sympy.abc import x, y # Import x and y directly for easier use

# Import optimization functions
from modules.analytical_methods import single_variable_optimization, multivariable_optimization, constrained_optimization
from modules.numerical_methods import optimize_single_variable, optimize_multivariable, optimize_constrained

# Set page configuration
st.set_page_config(page_title="Optimization Methods Explorer", layout="wide")

st.title("Optimization Methods Explorer")
st.markdown("""
This app demonstrates various optimization techniques with interactive visualizations:
- **Analytical Methods**: Symbolic solutions using calculus
- **Numerical Methods**: Iterative approaches like gradient descent and Newton's method
""")

# Main navigation
st.sidebar.header("Choose Method Type")
method_type = st.sidebar.radio(
    "Method Type",
    ["Analytical Methods", "Numerical Methods"]
)

# Define symbolic infinity range for periodic functions
int_range = (-oo, oo)  # Use symbolic infinity

# Define commonly used functions for demo examples
common_single_var = {
    "x³ - 3x² + 4": "x**3 - 3*x**2 + 4",
    "-x² + 4x": "-x**2 + 4*x",
    "sin(x) + cos(x)": "sin(x) + cos(x)",
    "x⁴ - 5x² + 4": "x**4 - 5*x**2 + 4"
}

common_multivar = {
    "x² + y² - 4x - 6y": "x**2 + y**2 - 4*x - 6*y",
    "-x² - y² + 4x + 6y": "-x**2 - y**2 + 4*x + 6*y",
    "x² + xy + y²": "x**2 + x*y + y**2",
    "sin(x) + cos(y)": "sin(x) + cos(y)"
}

common_obj_functions = {
    "xy": "x*y",
    "x² + y²": "x**2 + y**2",
    "x² - y²": "x**2 - y**2", 
    "e^x + e^y": "exp(x) + exp(y)"
}

common_constraints = {
    "x + y = 10": "x + y - 10",
    "x² + y² = 1": "x**2 + y**2 - 1",
    "x - 2y = 0": "x - 2*y",
    "x/y = 2": "x - 2*y"
}

# Numerical optimization common functions
num_common_single_var = {
    "Quadratic: (x - 2)²": "(x - 2)**2",
    "Cubic: x³ - 3x² + 4": "x**3 - 3*x**2 + 4",
    "Trigonometric: sin(x) + cos(x)": "sin(x) + cos(x)",
    "Quartic: x⁴ - 5x² + 4": "x**4 - 5*x**2 + 4"
}

num_common_multivar = {
    "Quadratic: x² + y² - 4x - 6y": "x**2 + y**2 - 4*x - 6*y",
    "Rosenbrock: (1-x)² + 100(y-x²)²": "(1-x)**2 + 100*(y-x**2)**2",
    "Bowl: x² + 2y²": "x**2 + 2*y**2",
    "Trigonometric: sin(x) + cos(y)": "sin(x) + cos(y)"
}

num_common_obj_functions = {
    "Product: xy": "x*y",
    "Sum of Squares: x² + y²": "x**2 + y**2",
    "Rosenbrock: (1-x)² + 100(y-x²)²": "(1-x)**2 + 100*(y-x**2)**2", 
    "Exponential: e^x + e^y": "exp(x) + exp(y)"
}

num_common_constraints = {
    "Linear: x + y = 10": "x + y - 10",
    "Circle: x² + y² = 1": "x**2 + y**2 - 1",
    "Line: x - 2y = 0": "x - 2*y",
    "Hyperbola: xy = 1": "x*y - 1"
}

if method_type == "Analytical Methods":
    st.sidebar.header("Choose Analytical Method")
    optimization_type = st.sidebar.selectbox(
        "Select Method",
        ["Single-variable Optimization", "Multivariable Optimization", "Constrained Optimization"]
    )
    
    if optimization_type == "Single-variable Optimization":
        st.header("Single-variable Optimization (Analytical)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Allow selection from common functions or custom input
            function_choice = st.radio("Function input method:", 
                                     ["Select from examples", "Enter custom function"])
            
            if function_choice == "Select from examples":
                selected_function = st.selectbox("Select a function:", 
                                               list(common_single_var.keys()),
                                               format_func=lambda x: f"f(x) = {x}")
                expr_str = common_single_var[selected_function]
                st.text_input("Function expression:", value=expr_str, disabled=True)
            else:
                expr_str = st.text_input("Enter function f(x):", value="x**3 - 3*x**2 + 4", 
                                       help="Use SymPy syntax, e.g., x**2 for x², sin(x) for sine")
        
        with col2:
            mode = st.radio("Optimization mode:", ('minimize', 'maximize'), index=0)
            st.markdown("### Variable")
            st.latex("x \\in \\mathbb{R}")
    
        if st.button("Optimize Single Variable", type="primary"):
            try:
                with st.spinner('Finding optimal point...'):
                    expr = sympify(expr_str)
                    # Check if it's a periodic function
                    is_periodic = any(func_name in expr_str for func_name in ['sin', 'cos', 'tan'])
                    
                    # Use preserve_symbolic=True for periodic functions
                    result = single_variable_optimization(expr, mode=mode, int_range=int_range, preserve_symbolic=True)
                    
                    # Even if a symbolic solution was returned, only display the numeric result
                    if len(result) == 4:  # Contains symbolic solution
                        best_point, best_val, fig, _ = result
                        
                        # Only display numeric solutions
                        st.success(f"Optimum point: x = {best_point:.4f}")
                        st.success(f"Optimum value: f(x) = {best_val:.4f}")
                        
                        # Display the LaTeX expression
                        st.markdown(f"### Function: $f(x) = {expr}$")
                        st.markdown(f"### {mode.capitalize()}d at $x = {best_point:.4f}$")
                        
                        # Display the figure if available
                        if fig:
                            st.pyplot(fig)
                    else:
                        best_point, best_val, fig = result
                        
                        st.success(f"Optimum point: x = {best_point:.4f}")
                        st.success(f"Optimum value: f(x) = {best_val:.4f}")
                        
                        # Display the LaTeX expression
                        st.markdown(f"### Function: $f(x) = {expr}$")
                        st.markdown(f"### {mode.capitalize()}d at $x = {best_point:.4f}$")
                        
                        # Display the figure
                        st.pyplot(fig)
                        
            except SympifyError:
                st.error("Invalid function expression. Please use valid SymPy syntax (e.g., x**2 + 3*x - 1).")
            except ValueError as e:
                st.error(f"Optimization Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
    
    elif optimization_type == "Multivariable Optimization":
        st.header("Multivariable Optimization (Analytical)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            function_choice = st.radio("Function input method:", 
                                     ["Select from examples", "Enter custom function"])
            
            if function_choice == "Select from examples":
                selected_function = st.selectbox("Select a function:", 
                                               list(common_multivar.keys()),
                                               format_func=lambda x: f"f(x,y) = {x}")
                expr_str = common_multivar[selected_function]
                st.text_input("Function expression:", value=expr_str, disabled=True)
            else:
                expr_str = st.text_input("Enter function f(x, y):", value="x**2 + y**2 - 4*x - 6*y",
                                       help="Use SymPy syntax, e.g., x**2 + y**2 for x² + y²")
        
        with col2:
            mode = st.radio("Optimization mode:", ('minimize', 'maximize'), index=0)
            st.markdown("### Variables")
            st.latex("x, y \\in \\mathbb{R}")
    
        if st.button("Optimize Multivariable", type="primary"):
            try:
                with st.spinner('Finding optimal point...'):
                    expr = sympify(expr_str)
                    result = multivariable_optimization(expr, mode=mode, int_range=int_range)
                    
                    # Even if a symbolic solution was returned, only display the numeric result
                    if isinstance(result, tuple) and len(result) == 4:  # Contains symbolic solution
                        best_point, best_val, fig, _ = result
                        
                        # Only display numeric solutions
                        st.success(f"Optimum point: (x, y) = ({best_point[0]:.4f}, {best_point[1]:.4f})")
                        st.success(f"Optimum value: f(x, y) = {best_val:.4f}")
                        
                        # Display the LaTeX expression
                        st.markdown(f"### Function: $f(x,y) = {expr}$")
                        st.markdown(f"### {mode.capitalize()}d at $(x,y) = ({best_point[0]:.4f}, {best_point[1]:.4f})$")
                        
                        # Display the figure if available
                        if fig:
                            st.pyplot(fig)
                    else:
                        best_point, best_val, fig = result
                        
                        st.success(f"Optimum point: (x, y) = ({best_point[0]:.4f}, {best_point[1]:.4f})")
                        st.success(f"Optimum value: f(x, y) = {best_val:.4f}")
                        
                        # Display the LaTeX expression
                        st.markdown(f"### Function: $f(x,y) = {expr}$")
                        st.markdown(f"### {mode.capitalize()}d at $(x,y) = ({best_point[0]:.4f}, {best_point[1]:.4f})$")
                        
                        # Display the figure
                        st.pyplot(fig)
                
            except SympifyError:
                st.error("Invalid function expression. Please use valid SymPy syntax (e.g., x**2 + y**2).")
            except ValueError as e:
                st.error(f"Optimization Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
    
    elif optimization_type == "Constrained Optimization":
        st.header("Constrained Optimization (Analytical)")
        
        st.markdown("Optimize $f(x,y)$ subject to constraint $g(x,y) = 0$ using Lagrange multipliers")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Objective function selection
            obj_function_choice = st.radio("Objective function input method:", 
                                         ["Select from examples", "Enter custom function"],
                                         key="obj_function_choice")
            
            if obj_function_choice == "Select from examples":
                selected_obj_function = st.selectbox("Select objective function:", 
                                                   list(common_obj_functions.keys()),
                                                   format_func=lambda x: f"f(x,y) = {x}")
                expr_str = common_obj_functions[selected_obj_function]
                st.text_input("Objective function expression:", value=expr_str, disabled=True)
            else:
                expr_str = st.text_input("Enter objective function f(x, y):", value="x*y",
                                       help="Use SymPy syntax, e.g., x*y for xy")
            
            # Constraint selection
            constraint_choice = st.radio("Constraint input method:", 
                                       ["Select from examples", "Enter custom constraint"],
                                       key="constraint_choice")
            
            if constraint_choice == "Select from examples":
                selected_constraint = st.selectbox("Select constraint:", 
                                                 list(common_constraints.keys()),
                                                 format_func=lambda x: f"g(x,y): {x}")
                constraint_str = common_constraints[selected_constraint]
                st.text_input("Constraint expression:", value=constraint_str, disabled=True)
            else:
                constraint_str = st.text_input("Enter constraint g(x, y) = 0:", value="x + y - 10",
                                             help="Enter in the form g(x,y) which will be set to zero")
        
        with col2:
            mode = st.radio("Optimization mode:", ('minimize', 'maximize'), index=0)
            st.markdown("### Variables")
            st.latex("x, y \\in \\mathbb{R}")
            st.markdown("### Method")
            st.markdown("Lagrange Multipliers")
    
        if st.button("Optimize Constrained", type="primary"):
            try:
                with st.spinner('Finding optimal point with constraint...'):
                    obj_expr = sympify(expr_str)
                    const_expr = sympify(constraint_str)
                    result = constrained_optimization(obj_expr, const_expr, mode=mode, int_range=int_range)
                    
                    # Even if a symbolic solution was returned, only display the numeric result
                    if isinstance(result, tuple) and len(result) == 4:  # Contains symbolic solution
                        opt_point, opt_val, fig, _ = result
                        
                        # Only display numeric solutions
                        st.success(f"Optimum point: (x, y) = ({opt_point[0]:.4f}, {opt_point[1]:.4f})")
                        st.success(f"Optimum value: f(x, y) = {opt_val:.4f}")
                        
                        # Display the LaTeX expressions
                        st.markdown(f"### Objective: $f(x,y) = {obj_expr}$")
                        st.markdown(f"### Constraint: $g(x,y) = {const_expr} = 0$")
                        st.markdown(f"### {mode.capitalize()}d at $(x,y) = ({opt_point[0]:.4f}, {opt_point[1]:.4f})$")
                        
                        # Display the figure if available
                        if fig:
                            st.pyplot(fig)
                    else:
                        opt_point, opt_val, fig = result
                        
                        st.success(f"Optimum point: (x, y) = ({opt_point[0]:.4f}, {opt_point[1]:.4f})")
                        st.success(f"Optimum value: f(x, y) = {opt_val:.4f}")
                        
                        # Display the LaTeX expressions
                        st.markdown(f"### Objective: $f(x,y) = {obj_expr}$")
                        st.markdown(f"### Constraint: $g(x,y) = {const_expr} = 0$")
                        st.markdown(f"### {mode.capitalize()}d at $(x,y) = ({opt_point[0]:.4f}, {opt_point[1]:.4f})$")
                        
                        # Display the figure
                        st.pyplot(fig)
                
            except SympifyError as e:
                st.error(f"Invalid function expression(s): {e}. Please use valid SymPy syntax.")
            except ValueError as e:
                st.error(f"Optimization Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

elif method_type == "Numerical Methods":
    st.header("Numerical Optimization Methods")
    st.markdown("""
    This section demonstrates numerical optimization techniques:
    - **Gradient Descent**: First-order method using function gradients
    - **Newton's Method**: Second-order method using both gradients and Hessians
    - **Penalty Method**: Handles constraints by adding penalty terms
    - **Barrier Method**: Handles constraints using log barriers
    """)

    # Sidebar options for numerical methods
    st.sidebar.header("Numerical Methods")
    numerical_type = st.sidebar.selectbox(
        "Select Numerical Method",
        ["Single-variable Numerical", "Multivariable Numerical", "Constrained Numerical"]
    )

    if numerical_type == "Single-variable Numerical":
        st.subheader("Single-variable Numerical Optimization")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Function selection
            function_choice = st.radio(
                "Function input method:", 
                ["Select from examples", "Enter custom function"],
                key="num_single_var_function_choice"
            )
            
            if function_choice == "Select from examples":
                selected_function = st.selectbox(
                    "Select a function:", 
                    list(num_common_single_var.keys()),
                    format_func=lambda x: f"f(x) = {x}",
                    key="num_single_var_select"
                )
                expr_str = num_common_single_var[selected_function]
                st.text_input("Function expression:", value=expr_str, disabled=True, key="num_single_var_expr_disabled")
            else:
                expr_str = st.text_input(
                    "Enter function f(x):", 
                    value="x**2 - 4*x + 4",
                    help="Use Python syntax: x**2 for x², sin(x), exp(x), etc.",
                    key="num_single_var_expr"
                )
            
            # Initial point selection
            x0 = st.number_input("Initial point (x₀):", value=0.0, step=0.5, key="num_single_var_x0")
            
            # Method selection
            method = st.radio(
                "Optimization method:", 
                ["newton", "gradient"],
                format_func=lambda x: "Newton's Method" if x == "newton" else "Gradient Descent",
                key="num_single_var_method"
            )
        
        with col2:
            mode = st.radio("Optimization mode:", ('minimize', 'maximize'), index=0, key="num_single_var_mode")
            st.markdown("### Variable")
            st.latex("x \\in \\mathbb{R}")
        
        if st.button("Run Numerical Optimization", type="primary", key="num_single_var_button"):
            try:
                with st.spinner('Running optimization...'):
                    x_opt, f_opt, fig, path = optimize_single_variable(
                        expr_str, 
                        mode=mode, 
                        x0=x0, 
                        method=method
                    )
                    
                    st.success(f"Optimum point: x = {x_opt:.6f}")
                    st.success(f"Optimum value: f(x) = {f_opt:.6f}")
                    
                    # Display function expression
                    st.markdown(f"### Function: $f(x) = {expr_str}$")
                    
                    # Display convergence info if path is available
                    if path:
                        st.info(f"Converged in {len(path)} iterations")
                    
                    # Display the figure
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Optimization Error: {str(e)}")

    elif numerical_type == "Multivariable Numerical":
        st.subheader("Multivariable Numerical Optimization")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Function selection
            function_choice = st.radio(
                "Function input method:", 
                ["Select from examples", "Enter custom function"],
                key="num_multivar_function_choice"
            )
            
            if function_choice == "Select from examples":
                selected_function = st.selectbox(
                    "Select a function:", 
                    list(num_common_multivar.keys()),
                    format_func=lambda x: f"f(x,y) = {x}",
                    key="num_multivar_select"
                )
                expr_str = num_common_multivar[selected_function]
                st.text_input("Function expression:", value=expr_str, disabled=True, key="num_multivar_expr_disabled")
            else:
                expr_str = st.text_input(
                    "Enter function f(x,y):", 
                    value="x**2 + y**2 - 2*x - 4*y",
                    help="Use Python syntax: x**2, y**2, sin(x), exp(y), etc.",
                    key="num_multivar_expr"
                )
            
            # Initial point selection
            col_x0, col_y0 = st.columns(2)
            with col_x0:
                x0 = st.number_input("Initial x₀:", value=0.0, step=0.5, key="num_multivar_x0")
            with col_y0:
                y0 = st.number_input("Initial y₀:", value=0.0, step=0.5, key="num_multivar_y0")
            
            # Method selection
            method = st.radio(
                "Optimization method:", 
                ["newton", "gradient", "scipy"],
                format_func=lambda x: {"newton": "Newton's Method", "gradient": "Gradient Descent", "scipy": "SciPy BFGS"}[x],
                key="num_multivar_method"
            )
        
        with col2:
            mode = st.radio("Optimization mode:", ('minimize', 'maximize'), index=0, key="num_multivar_mode")
            st.markdown("### Variables")
            st.latex("x, y \\in \\mathbb{R}")
        
        if st.button("Run Numerical Optimization", type="primary", key="num_multivar_button"):
            try:
                with st.spinner('Running optimization...'):
                    result = optimize_multivariable(
                        expr_str, 
                        mode=mode, 
                        x0=np.array([x0, y0]), 
                        method=method
                    )
                    
                    if len(result) == 5:  # With trajectory plot
                        x_opt, f_opt, fig, path, traj_fig = result
                        has_traj = True
                    else:
                        x_opt, f_opt, fig, path = result
                        has_traj = False
                    
                    st.success(f"Optimum point: (x, y) = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
                    st.success(f"Optimum value: f(x, y) = {f_opt:.6f}")
                    
                    # Display function expression
                    st.markdown(f"### Function: $f(x,y) = {expr_str}$")
                    
                    # Display convergence info if path is available
                    if path:
                        st.info(f"Converged in {len(path)} iterations")
                    
                    # Display figures
                    st.pyplot(fig)
                    
                    if has_traj:
                        st.subheader("Optimization Trajectory")
                        st.pyplot(traj_fig)
                    
            except Exception as e:
                st.error(f"Optimization Error: {str(e)}")

    elif numerical_type == "Constrained Numerical":
        st.subheader("Constrained Numerical Optimization")
        
        st.markdown("Optimize $f(x,y)$ subject to constraint $g(x,y) = 0$ using numerical methods")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Objective function selection
            obj_function_choice = st.radio(
                "Objective function input method:", 
                ["Select from examples", "Enter custom function"],
                key="num_obj_function_choice"
            )
            
            if obj_function_choice == "Select from examples":
                selected_obj_function = st.selectbox(
                    "Select objective function:", 
                    list(num_common_obj_functions.keys()),
                    format_func=lambda x: f"f(x,y) = {x}",
                    key="num_obj_select"
                )
                obj_expr_str = num_common_obj_functions[selected_obj_function]
                st.text_input("Objective function:", value=obj_expr_str, disabled=True, key="num_obj_expr_disabled")
            else:
                obj_expr_str = st.text_input(
                    "Enter objective function f(x,y):", 
                    value="x*y",
                    help="Use Python syntax: x*y for xy, x**2 for x², etc.",
                    key="num_obj_expr"
                )
            
            # Constraint selection
            constraint_choice = st.radio(
                "Constraint input method:", 
                ["Select from examples", "Enter custom constraint"],
                key="num_constraint_choice"
            )
            
            if constraint_choice == "Select from examples":
                selected_constraint = st.selectbox(
                    "Select constraint:", 
                    list(num_common_constraints.keys()),
                    format_func=lambda x: f"g(x,y): {x}",
                    key="num_constraint_select"
                )
                constraint_expr_str = num_common_constraints[selected_constraint]
                st.text_input("Constraint:", value=constraint_expr_str, disabled=True, key="num_constraint_expr_disabled")
            else:
                constraint_expr_str = st.text_input(
                    "Enter constraint g(x,y) = 0:", 
                    value="x + y - 10",
                    help="Enter in the form g(x,y) which will be set to zero",
                    key="num_constraint_expr"
                )
            
            # Initial point selection
            col_x0, col_y0 = st.columns(2)
            with col_x0:
                x0 = st.number_input("Initial x₀:", value=1.0, step=0.5, key="num_constrained_x0")
            with col_y0:
                y0 = st.number_input("Initial y₀:", value=1.0, step=0.5, key="num_constrained_y0")
            
            # Method selection
            method = st.radio(
                "Optimization method:", 
                ["penalty", "barrier"],
                format_func=lambda x: "Penalty Method" if x == "penalty" else "Barrier Method",
                key="num_constrained_method"
            )
            
            if method == "barrier":
                st.warning("Barrier method requires the initial point to be inside the feasible region!")
        
        with col2:
            mode = st.radio("Optimization mode:", ('minimize', 'maximize'), index=0, key="num_constrained_mode")
            st.markdown("### Variables")
            st.latex("x, y \\in \\mathbb{R}")
            st.markdown("### Constraint")
            st.latex("g(x,y) = 0")
        
        if st.button("Run Constrained Optimization", type="primary", key="num_constrained_button"):
            try:
                with st.spinner('Running constrained optimization...'):
                    result = optimize_constrained(
                        obj_expr_str, 
                        constraint_expr_str, 
                        mode=mode, 
                        x0=np.array([x0, y0]), 
                        method=method
                    )
                    
                    if len(result) == 5:  # With trajectory plot
                        x_opt, f_opt, fig, path, traj_fig = result
                        has_traj = True
                    else:
                        x_opt, f_opt, fig, path = result
                        has_traj = False
                    
                    st.success(f"Optimum point: (x, y) = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
                    st.success(f"Optimum value: f(x, y) = {f_opt:.6f}")
                    
                    # Display expressions
                    st.markdown(f"### Objective: $f(x,y) = {obj_expr_str}$")
                    st.markdown(f"### Constraint: $g(x,y) = {constraint_expr_str} = 0$")
                    
                    # Display convergence info if path is available
                    if path:
                        st.info(f"Converged in {len(path)} iterations")
                    
                    # Display figures
                    st.pyplot(fig)
                    
                    if has_traj:
                        st.subheader("Optimization Trajectory")
                        st.pyplot(traj_fig)
                    
            except Exception as e:
                st.error(f"Optimization Error: {str(e)}")

# Add common footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Select an optimization method type (Analytical or Numerical)
2. Choose the specific optimization method
3. Enter a function or select from examples
4. Select minimize or maximize
5. Click the optimize button
6. View results and visualizations
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Analytical Methods**: Use symbolic calculus to find exact solutions")
st.sidebar.markdown("**Numerical Methods**: Use iterative algorithms to find approximate solutions")
st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** All functions use syntax appropriate for the method")

# Add a footer with information about the app
st.markdown("---")
st.markdown("""
### About this App
This optimization explorer demonstrates both analytical (symbolic) and numerical methods for finding optimal points of functions.

**Analytical Methods**: Use calculus to find exact solutions by determining where derivatives equal zero.

**Numerical Methods**: Use iterative algorithms to approximate solutions, especially useful when:
- Exact solutions are difficult to find
- Functions have complex derivatives
- Problems involve constraints that are hard to handle analytically

The app shows visualizations of the functions and optimization processes to help understand how these methods work.
""")
