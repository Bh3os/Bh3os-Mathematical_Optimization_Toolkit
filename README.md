# Mathematical Optimization Toolkit

An interactive application for exploring various optimization methods with visualizations powered by Streamlit.

## Overview

The Mathematical Optimization Toolkit is a comprehensive educational tool that demonstrates both analytical and numerical optimization techniques. It provides interactive visualizations to help users understand how different optimization algorithms work.

## Features

- **Analytical Methods**:
  - Single-variable optimization
  - Multivariable optimization
  - Constrained optimization using Lagrange multipliers

- **Numerical Methods**:
  - Gradient descent
  - Newton's method
  - Constrained optimization techniques

- **Interactive Visualizations**:
  - Function plots with critical points
  - 3D surface and contour plots
  - Optimization trajectories
  - Step-by-step algorithmic progress

## Getting Started

### Prerequisites

Make sure you have Python installed on your system. This application requires Python 3.7+.

### Installation

1. Clone this repository or download the source code
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

You can run the application using the provided batch file:

```bash
run_app.bat
```

Or directly with Streamlit:

```bash
streamlit run app.py
```

## Usage Guide

1. Select a method type (Analytical or Numerical) from the sidebar
2. Choose the specific optimization method
3. Enter a custom function or select from the provided examples
4. Configure additional parameters (initial points, constraints, etc.)
5. Run the optimization process
6. Explore the visualization and results

## Example Functions

The application includes various example functions:

- **Single-variable**: Quadratic, cubic, trigonometric, and polynomial functions
- **Multivariable**: Quadratic surfaces, Rosenbrock function, bowl functions
- **Constrained**: Product maximization with budget constraints, distance minimization with equality constraints

## Project Structure

```
Optimization/
├── app.py                 # Main Streamlit application
├── modules/               # Core functionality modules
│   ├── analytical_methods.py
│   └── numerical_methods.py
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── run_app.bat            # Launcher script
```

## Dependencies

- Streamlit: Web application framework
- SymPy: Symbolic mathematics
- NumPy & SciPy: Numerical computing and optimization
- Matplotlib & Plotly: Data visualization

## License

[MIT License](LICENSE)

## Acknowledgments

- This toolkit was created as an educational resource for understanding optimization techniques
- Inspired by classical optimization problems in mathematics and engineering
