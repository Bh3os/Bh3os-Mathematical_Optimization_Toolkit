import streamlit as st
from sympy import sympify, SympifyError, pi, sqrt
from sympy.abc import x, y # Import x and y directly for easier use

# Import the optimization functions from the module
from enhanced_Analytical_Methods import single_variable_optimization, multivariable_optimization, constrained_optimization

st.set_page_config(page_title="Optimization Methods Explorer", layout="wide")

st.title("Optimization Methods Explorer")
st.markdown("""
This app demonstrates various optimization techniques with interactive visualizations:
- **Single-variable**: Find minimum/maximum points for functions of one variable
- **Multivariable**: Find minimum/maximum points for functions of two variables
- **Constrained**: Find optimal points subject to constraints using Lagrange multipliers
""")

st.sidebar.header("Choose Optimization Type")
optimization_type = st.sidebar.selectbox(
    "Select Method",
    ["Single-variable Optimization", "Multivariable Optimization", "Constrained Optimization"]
)

# Advanced settings header
st.sidebar.header("Advanced Settings")
st.sidebar.info("This application handles periodic functions (like sine and cosine) by finding the optimal points within a practical range.")

# Define symbolic infinity range for periodic functions - this will be used internally
# We'll use a fixed range for performance reasons but wide enough to catch most patterns
from sympy import oo  # Import sympy's infinity
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

if optimization_type == "Single-variable Optimization":
    st.header("Single-variable Optimization")
    
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
    st.header("Multivariable Optimization")
    
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
    st.header("Constrained Optimization")
    
    st.markdown("Optimize $f(x,y)$ subject to constraint $g(x,y) = 0$")
    
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

st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Select an optimization method
2. Enter a function or choose from examples
3. Select minimize or maximize
4. Click the optimize button
5. View results and visualization
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** All functions use SymPy syntax.")
st.sidebar.markdown("**Examples:**")
st.sidebar.markdown("- x^2 is written as `x**2`")
st.sidebar.markdown("- √x is written as `sqrt(x)`")
st.sidebar.markdown("- sin(x) is written as `sin(x)`")
st.sidebar.markdown("- e^x is written as `exp(x)`")
