"""
Enhanced Streamlit UI for Comprehensive Structural Analysis

This application provides a complete interface for structural analysis using the
Stiffness Matrix Method, supporting both frames and trusses with comprehensive
analysis capabilities.

Based on enhanced_structural_analysis.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from enhanced_structural_analysis import (
    StructuralAnalyzer, Node, Element, Load,
    example_cantilever_beam, example_simple_truss, example_indeterminate_frame
)

def format_number(value, decimal_places=3):
    """Format a number to display in decimal format instead of scientific notation."""
    if abs(value) < 1e-10:
        return "0"
    elif abs(value) >= 1e6:
        return f"{value:,.{max(0, decimal_places-3)}f}"
    elif abs(value) >= 1000:
        return f"{value:,.{max(1, decimal_places-2)}f}"
    elif abs(value) >= 1:
        return f"{value:.{decimal_places}f}"
    else:
        return f"{value:.{decimal_places+3}f}"

# Set page configuration
st.set_page_config(
    page_title="Enhanced Structural Analysis",
    page_icon="🏗️",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        text-align: center !important;
    }
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #0D47A1 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #0D47A1 !important;
    }
    .info-box {
        background-color: #E3F2FD !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    .result-box {
        background-color: #E8F5E9 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    .warning-box {
        background-color: #FFF3E0 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin-bottom: 1rem !important;
        border-left: 4px solid #FF9800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Enhanced Structural Analysis System</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
This comprehensive tool performs structural analysis of determinate/indeterminate frames and trusses using the <strong>Stiffness Matrix Method</strong>:
<ul>
    <li><strong>Frame Elements</strong>: Support axial, shear, and moment (6 DOF per element)</li>
    <li><strong>Truss Elements</strong>: Axial forces only (2 DOF per element)</li>
    <li><strong>Multiple Nodes</strong>: Define complex structures with any number of nodes</li>
    <li><strong>Various Supports</strong>: Pin, roller, fixed supports</li>
    <li><strong>Load Types</strong>: Point loads in any direction</li>
    <li><strong>Analysis Results</strong>: Displacements, reactions, internal forces</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = StructuralAnalyzer()
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Create main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Structure Definition", "Analysis & Results", "Examples", "Validation", "Theory"])

# Tab 1: Structure Definition
with tab1:
    st.markdown("<h2 class='section-header'>Structure Definition</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Nodes")
        
        with st.expander("Add Node", expanded=True):
            node_id = st.number_input("Node ID", min_value=0, value=0, step=1)
            node_x = st.number_input("X Coordinate (m)", value=0.0, step=0.1)
            node_y = st.number_input("Y Coordinate (m)", value=0.0, step=0.1)
            
            st.write("Support Conditions:")
            col_ux, col_uy, col_theta = st.columns(3)
            with col_ux:
                restraint_x = st.checkbox("Fix u_x", key=f"restraint_x_{node_id}")
            with col_uy:
                restraint_y = st.checkbox("Fix u_y", key=f"restraint_y_{node_id}")
            with col_theta:
                restraint_theta = st.checkbox("Fix θ", key=f"restraint_theta_{node_id}")
            
            if st.button("Add Node"):
                restraints = [restraint_x, restraint_y, restraint_theta]
                st.session_state.analyzer.add_node(node_id, node_x, node_y, restraints)
                st.success(f"Node {node_id} added successfully!")
                st.session_state.analysis_complete = False
        
        # Display current nodes
        if st.session_state.analyzer.nodes:
            st.write("**Current Nodes:**")
            nodes_data = []
            for node_id, node in st.session_state.analyzer.nodes.items():
                support_type = "Free"
                if all(node.restraints):
                    support_type = "Fixed"
                elif node.restraints[0] and node.restraints[1]:
                    support_type = "Pin"
                elif node.restraints[1]:
                    support_type = "Roller"
                    
                nodes_data.append({
                    "ID": node_id,
                    "X (m)": node.x,
                    "Y (m)": node.y,
                    "Support": support_type
                })
            st.dataframe(pd.DataFrame(nodes_data), use_container_width=True)
    
    with col2:
        st.subheader("Elements")
        
        with st.expander("Add Element", expanded=True):
            elem_id = st.number_input("Element ID", min_value=0, value=0, step=1)
            
            if st.session_state.analyzer.nodes:
                available_nodes = list(st.session_state.analyzer.nodes.keys())
                node_i = st.selectbox("Start Node", available_nodes)
                node_j = st.selectbox("End Node", available_nodes)
            else:
                st.warning("Please add nodes first")
                node_i = node_j = 0
            
            element_type = st.selectbox("Element Type", ["frame", "truss"])
            
            col_e, col_a = st.columns(2)
            with col_e:
                E = st.number_input("Young's Modulus E (kN/m²)", value=200000.0, min_value=1.0)
            with col_a:
                A = st.number_input("Cross-sectional Area A (m²)", value=0.01, min_value=0.0001)
            
            if element_type == "frame":
                I = st.number_input("Moment of Inertia I (m⁴)", value=0.0001, min_value=0.000001)
            else:
                I = 0.0
                st.info("Truss elements don't require moment of inertia")
            
            if st.button("Add Element"):
                if node_i != node_j:
                    st.session_state.analyzer.add_element(elem_id, node_i, node_j, E, A, I, element_type)
                    st.success(f"Element {elem_id} added successfully!")
                    st.session_state.analysis_complete = False
                else:
                    st.error("Start and end nodes must be different")
        
        # Display current elements
        if st.session_state.analyzer.elements:
            st.write("**Current Elements:**")
            elements_data = []
            for elem_id, elem in st.session_state.analyzer.elements.items():
                length, angle = st.session_state.analyzer.get_element_length_angle(elem)
                elements_data.append({
                    "ID": elem_id,
                    "Type": elem.element_type.title(),
                    "Nodes": f"{elem.node_i}-{elem.node_j}",
                    "Length (m)": f"{length:.3f}",
                    "Angle (°)": f"{np.degrees(angle):.1f}",
                    "E (kN/m²)": f"{elem.E:,.0f}",
                    "A (m²)": f"{elem.A:.4f}"
                })
            st.dataframe(pd.DataFrame(elements_data), use_container_width=True)
    
    # Loads section
    st.subheader("Loads")
    
    with st.expander("Add Load", expanded=True):
        if st.session_state.analyzer.nodes:
            load_node = st.selectbox("Node", list(st.session_state.analyzer.nodes.keys()))
            load_dof = st.selectbox("Direction", [0, 1, 2], format_func=lambda x: ["X (Horizontal)", "Y (Vertical)", "Moment"][x])
            load_magnitude = st.number_input("Magnitude", value=0.0, step=1.0)
            
            if st.button("Add Load"):
                st.session_state.analyzer.add_load(load_node, load_dof, load_magnitude)
                st.success(f"Load added to node {load_node}!")
                st.session_state.analysis_complete = False
        else:
            st.warning("Please add nodes first")
    
    # Display current loads
    if st.session_state.analyzer.loads:
        st.write("**Current Loads:**")
        loads_data = []
        for i, load in enumerate(st.session_state.analyzer.loads):
            direction = ["X (Horizontal)", "Y (Vertical)", "Moment"][load.dof]
            loads_data.append({
                "Load #": i+1,
                "Node": load.node_id,
                "Direction": direction,
                "Magnitude": f"{load.magnitude:.2f}"
            })
        st.dataframe(pd.DataFrame(loads_data), use_container_width=True)
    
    # Clear structure button
    if st.button("Clear All", type="secondary"):
        st.session_state.analyzer = StructuralAnalyzer()
        st.session_state.analysis_complete = False
        st.success("Structure cleared!")
        st.rerun()

# Tab 2: Analysis & Results
with tab2:
    st.markdown("<h2 class='section-header'>Analysis & Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.analyzer.nodes:
        st.warning("Please define a structure first in the 'Structure Definition' tab.")
    elif not st.session_state.analyzer.elements:
        st.warning("Please add at least one element to the structure.")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Run Analysis", type="primary"):
                try:
                    st.session_state.analyzer.solve()
                    st.session_state.analysis_complete = True
                    st.success("Analysis completed successfully!")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.session_state.analysis_complete = False
        
        if st.session_state.analysis_complete:
            # Display results
            results = st.session_state.analyzer.get_results_summary()
            
            st.subheader("Analysis Results")
            
            # Node results
            st.write("**Node Displacements and Reactions:**")
            node_results_data = []
            for node_id, node_data in results['nodes'].items():
                node_results_data.append({
                    "Node": node_id,
                    "u_x (m)": format_number(node_data['displacements']['u_x'], 6),
                    "u_y (m)": format_number(node_data['displacements']['u_y'], 6),
                    "θ (rad)": format_number(node_data['displacements']['theta'], 6),
                    "R_x (kN)": format_number(node_data['reactions']['R_x'], 3),
                    "R_y (kN)": format_number(node_data['reactions']['R_y'], 3),
                    "M_z (kN⋅m)": format_number(node_data['reactions']['M_z'], 3)
                })
            st.dataframe(pd.DataFrame(node_results_data), use_container_width=True)
            
            # Element forces
            if results['elements']:
                st.write("**Element Internal Forces:**")
                element_results_data = []
                for elem_id, elem_forces in results['elements'].items():
                    element_results_data.extend([
                        {
                            "Element": elem_id,
                            "Node": "i",
                            "Fx (kN)": format_number(elem_forces['node_i']['Fx'], 3),
                            "Fy (kN)": format_number(elem_forces['node_i']['Fy'], 3),
                            "Mz (kN⋅m)": format_number(elem_forces['node_i']['Mz'], 3)
                        },
                        {
                            "Element": elem_id,
                            "Node": "j",
                            "Fx (kN)": format_number(elem_forces['node_j']['Fx'], 3),
                            "Fy (kN)": format_number(elem_forces['node_j']['Fy'], 3),
                            "Mz (kN⋅m)": format_number(elem_forces['node_j']['Mz'], 3)
                        }
                    ])
                st.dataframe(pd.DataFrame(element_results_data), use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Displacement", f"{results['max_displacement']:.6f} m")
            with col2:
                st.metric("Max Reaction", f"{results['max_reaction']:.3f} kN")
            with col3:
                st.metric("Total Nodes", len(st.session_state.analyzer.nodes))
            with col4:
                st.metric("Total Elements", len(st.session_state.analyzer.elements))
            
            # Visualization
            st.subheader("Structure Visualization")
            
            col1, col2 = st.columns(2)
            with col1:
                show_deformed = st.checkbox("Show Deformed Shape", value=True)
            with col2:
                if show_deformed:
                    scale_factor = st.slider("Deformation Scale Factor", 1, 1000, 100)
                else:
                    scale_factor = 100
            
            try:
                fig = st.session_state.analyzer.plot_structure(show_deformed, scale_factor)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
            
            # Export results
            st.subheader("Export Results")
            if st.button("Export to JSON"):
                filename = "analysis_results.json"
                st.session_state.analyzer.export_results(filename)
                st.success(f"Results exported to {filename}")
                
                # Show download button
                with open(filename, 'r') as f:
                    st.download_button(
                        label="Download Results",
                        data=f.read(),
                        file_name=filename,
                        mime="application/json"
                    )

# Tab 3: Examples
with tab3:
    st.markdown("<h2 class='section-header'>Example Problems</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Load predefined example structures to understand the capabilities of the analysis system.
    These examples include validation against analytical solutions.
    </div>
    """, unsafe_allow_html=True)
    
    example_tabs = st.tabs(["Cantilever Beam", "Simple Truss", "Indeterminate Frame"])
    
    with example_tabs[0]:
        st.subheader("Cantilever Beam Example")
        st.write("""
        **Problem Description:**
        - 4m cantilever beam with fixed support at left end
        - 10 kN downward load at free end
        - Steel beam: E = 200 GPa, A = 0.01 m², I = 8.33×10⁻⁶ m⁴
        """)
        
        if st.button("Load Cantilever Example"):
            st.session_state.analyzer = example_cantilever_beam()
            st.session_state.analysis_complete = True
            st.success("Cantilever beam example loaded and analyzed!")
            st.rerun()
    
    with example_tabs[1]:
        st.subheader("Simple Truss Example")
        st.write("""
        **Problem Description:**
        - Triangular truss with pin and roller supports
        - 20 kN downward load at apex
        - Steel members: E = 200 GPa, A = 0.01 m²
        """)
        
        if st.button("Load Truss Example"):
            st.session_state.analyzer = example_simple_truss()
            st.session_state.analysis_complete = True
            st.success("Simple truss example loaded and analyzed!")
            st.rerun()
    
    with example_tabs[2]:
        st.subheader("Indeterminate Frame Example")
        st.write("""
        **Problem Description:**
        - Portal frame with fixed supports at both ends
        - Horizontal and vertical loads
        - Steel frame: E = 200 GPa, A = 0.02 m², I = 1.67×10⁻⁴ m⁴
        """)
        
        if st.button("Load Frame Example"):
            st.session_state.analyzer = example_indeterminate_frame()
            st.session_state.analysis_complete = True
            st.success("Indeterminate frame example loaded and analyzed!")
            st.rerun()

# Tab 4: Validation
with tab4:
    st.markdown("<h2 class='section-header'>Validation & Verification</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This section provides validation of the numerical results against analytical solutions
    and comparison with other structural analysis software.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.analysis_complete:
        st.subheader("Current Analysis Validation")
        
        # Check if this is a cantilever beam (for analytical comparison)
        if (len(st.session_state.analyzer.nodes) == 2 and 
            len(st.session_state.analyzer.elements) == 1 and
            len(st.session_state.analyzer.loads) == 1):
            
            # Get beam properties
            element = list(st.session_state.analyzer.elements.values())[0]
            load = st.session_state.analyzer.loads[0]
            
            if (element.element_type == 'frame' and 
                load.node_id == 1 and load.dof == 1):  # Tip load
                
                L, _ = st.session_state.analyzer.get_element_length_angle(element)
                P = abs(load.magnitude)
                E = element.E
                I = element.I
                
                # Analytical solutions
                analytical_deflection = -P * L**3 / (3 * E * I)
                analytical_rotation = -P * L**2 / (2 * E * I)
                
                # Numerical results
                numerical_deflection = st.session_state.analyzer.displacements[4]
                numerical_rotation = st.session_state.analyzer.displacements[5]
                
                # Calculate errors
                deflection_error = abs((numerical_deflection - analytical_deflection) / analytical_deflection) * 100
                rotation_error = abs((numerical_rotation - analytical_rotation) / analytical_rotation) * 100
                
                st.write("**Cantilever Beam Validation:**")
                
                validation_data = [
                    {
                        "Parameter": "Tip Deflection (m)",
                        "Analytical": f"{analytical_deflection:.6f}",
                        "Numerical": f"{numerical_deflection:.6f}",
                        "Error (%)": f"{deflection_error:.4f}"
                    },
                    {
                        "Parameter": "Tip Rotation (rad)",
                        "Analytical": f"{analytical_rotation:.6f}",
                        "Numerical": f"{numerical_rotation:.6f}",
                        "Error (%)": f"{rotation_error:.4f}"
                    }
                ]
                
                st.dataframe(pd.DataFrame(validation_data), use_container_width=True)
                
                if deflection_error < 0.01 and rotation_error < 0.01:
                    st.success("✅ Excellent agreement with analytical solution (< 0.01% error)")
                elif deflection_error < 0.1 and rotation_error < 0.1:
                    st.success("✅ Good agreement with analytical solution (< 0.1% error)")
                else:
                    st.warning("⚠️ Moderate agreement with analytical solution")
        else:
            st.info("Load a cantilever beam example for analytical validation comparison.")
    else:
        st.warning("Please run an analysis first to enable validation.")
    
    st.subheader("Validation Guidelines")
    st.markdown("""
    **Validation Methods:**
    1. **Analytical Solutions**: Compare with hand calculations for simple structures
    2. **Commercial Software**: Verify against SAP2000, ETABS, or ANSYS results
    3. **Literature Examples**: Use published benchmark problems
    4. **Equilibrium Check**: Verify force and moment equilibrium
    
    **Acceptable Error Ranges:**
    - Displacements: < 0.1% for simple structures, < 1% for complex structures
    - Reactions: Should satisfy equilibrium (sum of forces = applied loads)
    - Internal Forces: < 0.5% for determinate structures
    """)

# Tab 5: Theory
with tab5:
    st.markdown("<h2 class='section-header'>Theoretical Background</h2>", unsafe_allow_html=True)
    
    theory_tabs = st.tabs(["Stiffness Method", "Element Matrices", "Assembly Process", "Solution Procedure"])
    
    with theory_tabs[0]:
        st.markdown("""
        ### Direct Stiffness Method
        
        The Direct Stiffness Method is a matrix method for structural analysis based on:
        
        **Fundamental Equation:**
        ```
        [K]{u} = {F}
        ```
        
        Where:
        - **[K]** = Global stiffness matrix
        - **{u}** = Global displacement vector
        - **{F}** = Global force vector
        
        **Key Assumptions:**
        1. Linear elastic material behavior
        2. Small displacements and rotations
        3. Plane sections remain plane (Bernoulli-Euler beam theory)
        4. No local buckling or instability
        
        **Degrees of Freedom:**
        - **Frame Elements**: 6 DOF (3 per node: u_x, u_y, θ_z)
        - **Truss Elements**: 4 DOF (2 per node: u_x, u_y)
        """)
    
    with theory_tabs[1]:
        st.markdown("""
        ### Element Stiffness Matrices
        
        #### Frame Element (Local Coordinates)
        ```
        [k] = 
        ⎡  EA/L      0        0      -EA/L      0        0    ⎤
        ⎢   0    12EI/L³   6EI/L²     0    -12EI/L³   6EI/L² ⎥
        ⎢   0     6EI/L²    4EI/L     0     -6EI/L²    2EI/L  ⎥
        ⎢ -EA/L     0        0       EA/L      0        0    ⎥
        ⎢   0   -12EI/L³  -6EI/L²     0     12EI/L³  -6EI/L² ⎥
        ⎣   0     6EI/L²    2EI/L     0     -6EI/L²    4EI/L  ⎦
        ```
        
        #### Truss Element (Local Coordinates)
        ```
        [k] = (EA/L) ⎡  1   0  -1   0 ⎤
                     ⎢  0   0   0   0 ⎥
                     ⎢ -1   0   1   0 ⎥
                     ⎣  0   0   0   0 ⎦
        ```
        
        **Transformation to Global Coordinates:**
        ```
        [K] = [T]ᵀ[k][T]
        ```
        
        Where [T] is the transformation matrix based on element orientation.
        """)
    
    with theory_tabs[2]:
        st.markdown("""
        ### Assembly Process
        
        **Step 1: Initialize Global Matrix**
        - Size: (3×n) × (3×n) for n nodes
        - Initialize with zeros
        
        **Step 2: Element Contribution**
        For each element:
        1. Calculate element stiffness matrix [k_e]
        2. Transform to global coordinates [K_e]
        3. Identify global DOF indices
        4. Add [K_e] to global [K] at appropriate locations
        
        **Step 3: Assembly Rule**
        ```
        K[i,j] += k_e[local_i, local_j]
        ```
        
        Where i,j are global DOF indices corresponding to local DOFs.
        
        **Properties of Global Stiffness Matrix:**
        - Symmetric: K[i,j] = K[j,i]
        - Positive definite (for stable structures)
        - Sparse (many zero elements)
        - Singular before applying boundary conditions
        """)
    
    with theory_tabs[3]:
        st.markdown("""
        ### Solution Procedure
        
        **Step 1: Apply Boundary Conditions**
        Partition the system:
        ```
        ⎡ Kff  Kfs ⎤ ⎧ uf ⎫   ⎧ Ff ⎫
        ⎣ Ksf  Kss ⎦ ⎩ us ⎭ = ⎩ Fs ⎭
        ```
        
        Where:
        - f = free DOFs
        - s = supported (fixed) DOFs
        - us = 0 (known displacements)
        
        **Step 2: Solve for Free Displacements**
        ```
        [Kff]{uf} = {Ff}
        {uf} = [Kff]⁻¹{Ff}
        ```
        
        **Step 3: Calculate Reactions**
        ```
        {Rs} = [Kss]{us} + [Ksf]{uf} = [Ksf]{uf}
        ```
        
        **Step 4: Calculate Internal Forces**
        For each element, calculate internal forces using:
        ```
        {f} = [k]{d}
        ```
        Where:
        - [k] is the element stiffness matrix
        - {d} is the element displacement vector
        - {f} is the element force vector
        
        **Step 5: Post-processing**
        - Calculate axial forces, shear forces, and bending moments
        - Generate deformed shape
        - Calculate stress and strain distributions
        
        ### Validation
        
        The stiffness matrix method can be validated by comparing results with:
        - Analytical solutions for simple cases
        - Experimental data
        - Commercial software
        
        For example, for a cantilever beam with a point load P at the free end:
        - Analytical tip deflection: δ = PL³/(3EI)
        - Analytical tip rotation: θ = PL²/(2EI)
        
        ### Limitations
        
        - Linear elastic analysis only
        - Small deformation theory
        - No material nonlinearity
        - No geometric nonlinearity
        - Static analysis only
        """)