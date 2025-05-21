import streamlit as st
from sympy import sympify, SympifyError, pi, sqrt
from sympy.abc import x, y

# Import the optimization functions
from enhanced_Analytical_Methods import single_variable_optimization, multivariable_optimization, constrained_optimization
from numerical_streamlit import add_numerical_methods_to_streamlit

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

if method_type == "Analytical Methods":
    st.sidebar.header("Choose Analytical Method")
    optimization_type = st.sidebar.selectbox(
        "Select Method",
        ["Single-variable Optimization", "Multivariable Optimization", "Constrained Optimization"]
    )
    
    # Advanced settings header
    st.sidebar.header("Advanced Settings")
    st.sidebar.info("This application handles periodic functions (like sine and cosine) by finding the optimal points within a practical range.")
    
    # Define symbolic infinity range for periodic functions
    from sympy import oo
    int_range = (-oo, oo)
    
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
    # Invoke the numerical methods UI
    add_numerical_methods_to_streamlit()

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
