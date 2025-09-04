# ============================================================================
# ENHANCED STREAMLIT APP - NACA Airfoil Cp Predictor with Multiple Airfoil Analysis
# ============================================================================
# Save this as: enhanced_airfoil_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from scipy.interpolate import interp1d

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Neural Network Airfoil Cp Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL LOADING WITH CACHING
# ============================================================================

@st.cache_resource
def load_model():
    """
    Load the trained neural network model
    Uses Streamlit caching to load only once
    """
    try:
        # Try to load the best model from training
        model_path = 'best_airfoil_model.keras'
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            st.success("✅ Loaded best trained model successfully!")
            return model, True
        else:
            st.error("❌ Model file not found. Please ensure 'best_airfoil_model.keras' is in the same directory.")
            return None, False
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, False

# ============================================================================
# COORDINATE GENERATION FUNCTIONS
# ============================================================================

@st.cache_data
def generate_naca_coordinates(naca_code, num_points=200):
    """
    Generate NACA 4-digit airfoil coordinates
    Uses same function as training data generation
    """
    # Parse NACA code
    m = int(naca_code[0]) / 100.0    # Maximum camber
    p = int(naca_code[1]) / 10.0 if int(naca_code[1]) > 0 else 0.01  # Position of max camber
    t = int(naca_code[2:4]) / 100.0  # Maximum thickness
    
    # Generate x coordinates (cosine distribution)
    beta = np.linspace(0, np.pi, num_points)
    x = (1 - np.cos(beta)) / 2
    
    # Thickness distribution
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                  0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line
    if m == 0:  # Symmetric airfoil
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:  # Cambered airfoil
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        # Forward part (0 <= x <= p)
        forward_mask = x <= p
        yc[forward_mask] = (m / p**2) * (2 * p * x[forward_mask] - x[forward_mask]**2)
        dyc_dx[forward_mask] = (2 * m / p**2) * (p - x[forward_mask])
        
        # Aft part (p < x <= 1)
        aft_mask = x > p
        yc[aft_mask] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[aft_mask] - x[aft_mask]**2)
        dyc_dx[aft_mask] = (2 * m / (1 - p)**2) * (p - x[aft_mask])
    
    # Calculate surface coordinates
    theta = np.arctan(dyc_dx)
    
    # Upper surface
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    
    # Lower surface
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)
    
    # Combine coordinates (upper surface first, then lower surface reversed)
    x_coords = np.concatenate([x_upper, x_lower[::-1]])
    y_coords = np.concatenate([y_upper, y_lower[::-1]])
    
    return x_coords, y_coords

@st.cache_data
def scale_coordinates_for_prediction(x_coords, y_coords):
    """
    Scale coordinates to [0,1] format for neural network input
    Uses same scaling as training data
    """
    # Standardize to exactly 200 coordinate points
    target_points = 200
    
    # Calculate cumulative distance along airfoil
    distances = np.zeros(len(x_coords))
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        distances[i] = distances[i-1] + np.sqrt(dx*dx + dy*dy)
    
    total_length = distances[-1]
    target_distances = np.linspace(0, total_length, target_points)
    
    # Interpolate coordinates
    x_interp = interp1d(distances, x_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
    y_interp = interp1d(distances, y_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    x_resampled = x_interp(target_distances)
    y_resampled = y_interp(target_distances)
    
    # Scale to [0,1] range
    # X coordinates: normalize to [0,1]
    x_min, x_max = x_resampled.min(), x_resampled.max()
    if x_max > x_min:
        x_scaled = (x_resampled - x_min) / (x_max - x_min)
    else:
        x_scaled = x_resampled
    
    # Y coordinates: use global NACA scaling
    y_global_min = -0.15
    y_global_max = 0.15
    y_scaled = np.clip((y_resampled - y_global_min) / (y_global_max - y_global_min), 0, 1)
    
    # Create interleaved array for neural network: [x1, y1, x2, y2, ...]
    scaled_coords = np.zeros(target_points * 2)
    scaled_coords[0::2] = x_scaled  # Even indices: x
    scaled_coords[1::2] = y_scaled  # Odd indices: y
    
    return scaled_coords

# ============================================================================
# AIRCRAFT PRESET CONFIGURATIONS
# ============================================================================

def get_aircraft_presets():
    """
    Define preset aircraft configurations with typical NACA airfoils
    Based on real-world aircraft applications
    """
    presets = {
        "Custom (Manual Input)": {
            "description": "Design your own airfoil using the sliders",
            "camber": 2,
            "position": 4, 
            "thickness": 12,
            "naca": "2412",
            "details": "Use the parameter sliders to create your custom airfoil configuration."
        },
        
        "Commercial Passenger Aircraft": {
            "description": "Boeing 737, Airbus A320 - Cruise efficiency optimized",
            "camber": 2,
            "position": 4,
            "thickness": 15,
            "naca": "2415",
            "details": "Moderate camber for good lift-to-drag ratio at cruise conditions. Sufficient thickness for fuel storage and structural integrity. Used on aircraft like Boeing 737, Airbus A320."
        },
        
        "Business Jet": {
            "description": "Citation, Gulfstream - High speed performance",
            "camber": 1,
            "position": 3,
            "thickness": 12,
            "naca": "1312",
            "details": "Low camber and moderate thickness for high-speed cruise efficiency. Forward camber position for good pressure recovery. Typical of executive aircraft."
        },
        
        "Military Fighter Aircraft": {
            "description": "F-16, F/A-18 - High speed maneuverability",
            "camber": 0,
            "position": 4,  # Not used for symmetric
            "thickness": 9,
            "naca": "0009",
            "details": "Symmetric airfoil with thin section for minimal drag at high speeds. Zero camber provides identical performance inverted. Typical of fighter aircraft wing sections."
        },
        
        "General Aviation Training": {
            "description": "Cessna 172, Piper Cherokee - Stable flight characteristics",
            "camber": 2,
            "position": 4,
            "thickness": 12,
            "naca": "2412",
            "details": "Classic general aviation airfoil. Good stall characteristics, predictable handling, and adequate performance for training aircraft. Most common GA airfoil."
        },
        
        "Cargo Transport Aircraft": {
            "description": "C-130, Boeing 747F - Heavy load capability",
            "camber": 4,
            "position": 4,
            "thickness": 18,
            "naca": "4418",
            "details": "High camber for maximum lift capability. Thick section provides structural strength for heavy loads and large internal volume for cargo."
        },
        
        "Agricultural Aircraft": {
            "description": "Air Tractor, Thrush - Low speed, high lift",
            "camber": 5,
            "position": 5,
            "thickness": 20,
            "naca": "5520",
            "details": "High camber with aft position for maximum lift at low speeds. Thick section for structural strength and chemical tank volume. Optimized for low-altitude operations."
        },
        
        "Wind Turbine Blade": {
            "description": "Commercial wind turbine - Power extraction optimization",
            "camber": 6,
            "position": 4,
            "thickness": 25,
            "naca": "6425",
            "details": "Very high camber for maximum power extraction. Very thick section for structural durability in harsh weather. Designed for consistent power generation."
        },
        
        "Aerobatic Aircraft": {
            "description": "Extra 300, Edge 540 - Precision maneuvering",
            "camber": 0,
            "position": 4,  # Not used for symmetric
            "thickness": 12,
            "naca": "0012",
            "details": "Symmetric airfoil for identical performance upright and inverted. Moderate thickness for adequate strength during high-G maneuvers."
        },
        
        "Glider/Sailplane": {
            "description": "High-performance sailplane - Maximum L/D ratio",
            "camber": 2,
            "position": 3,
            "thickness": 15,
            "naca": "2315",
            "details": "Optimized for maximum lift-to-drag ratio. Forward camber position and moderate thickness for efficient soaring flight."
        }
    }
    
    return presets

# ============================================================================
# AIRFOIL CONFIGURATION FUNCTION FOR MULTIPLE ANALYSIS
# ============================================================================

def configure_airfoil(airfoil_key, container, aircraft_presets, default_preset="Custom (Manual Input)"):
    """
    Create airfoil configuration interface for single airfoil
    Returns: (naca_code, selected_preset, camber, position, thickness)
    """
    with container:
        st.markdown(f"### 🛩️ **Airfoil {airfoil_key.split('_')[1]}**")
        
        # Aircraft type selector
        preset_names = list(aircraft_presets.keys())
        selected_preset = st.selectbox(
            f"**Aircraft Type**",
            preset_names,
            index=preset_names.index(default_preset) if default_preset in preset_names else 0,
            key=f"preset_{airfoil_key}",
            help="Select a preset aircraft configuration or choose 'Custom' for manual input"
        )
        
        # Show preset information
        if selected_preset != "Custom (Manual Input)":
            preset_info = aircraft_presets[selected_preset]
            st.info(f"**NACA {preset_info['naca']}** - {preset_info['description']}")
        
        # Determine if we should use preset values
        if selected_preset == "Custom (Manual Input)":
            default_camber = 2
            default_position = 4
            default_thickness = 12
        else:
            preset_config = aircraft_presets[selected_preset]
            default_camber = preset_config['camber']
            default_position = preset_config['position']
            default_thickness = preset_config['thickness']
        
        # Parameter sliders
        camber = st.slider(
            "**Camber (M)**",
            min_value=0,
            max_value=7,
            value=default_camber,
            step=1,
            key=f"camber_{airfoil_key}",
            help="Maximum camber as percentage of chord (0-7%)"
        )
        
        # Position slider (only if camber > 0)
        if camber > 0:
            position = st.slider(
                "**Camber Position (P)**", 
                min_value=2,
                max_value=6,
                value=default_position,
                step=1,
                key=f"position_{airfoil_key}",
                help="Position of maximum camber (20-60% of chord)"
            )
        else:
            position = 4
            st.markdown("*Position not applicable for symmetric airfoils*")
        
        thickness = st.slider(
            "**Thickness (XX)**",
            min_value=6,
            max_value=30,
            value=default_thickness,
            step=1,
            key=f"thickness_{airfoil_key}",
            help="Maximum thickness as percentage of chord (6-30%)"
        )
        
        # Generate NACA code
        if camber == 0:
            naca_code = f"00{thickness:02d}"
        else:
            naca_code = f"{camber}{position}{thickness:02d}"
        
        # Show current configuration
        st.markdown(f"**Current: NACA {naca_code}**")
        
        return naca_code, selected_preset, camber, position, thickness

# ============================================================================
# SINGLE AIRFOIL ANALYSIS (ORIGINAL FUNCTIONALITY)
# ============================================================================

def single_airfoil_analysis(model):
    """Original single airfoil analysis functionality"""
    
    # Get preset configurations
    aircraft_presets = get_aircraft_presets()
    preset_names = list(aircraft_presets.keys())
    
    # ========================================================================
    # SIDEBAR: PRESET SELECTION + PARAMETER CONTROLS
    # ========================================================================
    
    st.sidebar.markdown("## 🎯 Aircraft Configuration Selection")
    
    # Aircraft type selector
    selected_preset = st.sidebar.selectbox(
        "**Choose Aircraft Type**",
        preset_names,
        index=0,  # Default to "Custom"
        help="Select a preset aircraft configuration or choose 'Custom' for manual input"
    )
    
    # Show preset information
    if selected_preset != "Custom (Manual Input)":
        preset_info = aircraft_presets[selected_preset]
        st.sidebar.markdown(f"### 📋 {selected_preset}")
        st.sidebar.info(f"**NACA {preset_info['naca']}**\n\n{preset_info['description']}")
        
        with st.sidebar.expander("ℹ️ Technical Details"):
            st.markdown(preset_info['details'])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ NACA Parameters")
    
    # Determine if we should use preset values or manual input
    if selected_preset == "Custom (Manual Input)":
        st.sidebar.markdown("*Use sliders below to design your airfoil:*")
        use_preset = False
        # Default values for custom
        default_camber = 2
        default_position = 4
        default_thickness = 12
    else:
        st.sidebar.markdown("*Preset values loaded (you can still adjust):*")
        use_preset = True
        preset_config = aircraft_presets[selected_preset]
        default_camber = preset_config['camber']
        default_position = preset_config['position']
        default_thickness = preset_config['thickness']
    
    # Parameter sliders with dynamic defaults
    camber = st.sidebar.slider(
        "**Camber (M)**",
        min_value=0,
        max_value=7,
        value=default_camber,
        step=1,
        help="Maximum camber as percentage of chord (0-7%)"
    )
    
    # Only show position slider if camber > 0
    if camber > 0:
        position = st.sidebar.slider(
            "**Camber Position (P)**", 
            min_value=2,
            max_value=6,
            value=default_position,
            step=1,
            help="Position of maximum camber (20-60% of chord)"
        )
    else:
        position = 4  # Default position for symmetric airfoils (not used)
        st.sidebar.markdown("*Camber position not applicable for symmetric airfoils*")
    
    thickness = st.sidebar.slider(
        "**Thickness (XX)**",
        min_value=6,
        max_value=30,
        value=default_thickness,
        step=1,
        help="Maximum thickness as percentage of chord (6-30%)"
    )
    
    # Generate NACA code
    if camber == 0:
        naca_code = f"00{thickness:02d}"
    else:
        naca_code = f"{camber}{position}{thickness:02d}"
    
    # Show current configuration in sidebar
    st.sidebar.markdown("### 🏷️ Current Configuration")
    st.sidebar.markdown(f"**NACA {naca_code}**")
    
    if use_preset and selected_preset != "Custom (Manual Input)":
        original_naca = aircraft_presets[selected_preset]['naca']
        if naca_code != original_naca:
            st.sidebar.warning(f"⚠️ Modified from preset NACA {original_naca}")
    
    # ========================================================================
    # MAIN AREA: CONFIGURATION DISPLAY AND ANALYSIS
    # ========================================================================
    
    # Display current airfoil parameters
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Aircraft Type", selected_preset.split()[0] if selected_preset != "Custom (Manual Input)" else "Custom")
    with col2:
        st.metric("NACA Code", naca_code)
    with col3:
        st.metric("Camber", f"{camber}%")
    with col4:
        st.metric("Position", f"{position*10}%" if camber > 0 else "N/A")
    with col5:
        st.metric("Thickness", f"{thickness}%")
    
    # Show aircraft application info
    if selected_preset != "Custom (Manual Input)":
        st.info(f"**{selected_preset}**: {aircraft_presets[selected_preset]['description']}")
    
    st.markdown("---")
    
    # Generate airfoil coordinates and perform analysis
    try:
        with st.spinner("🔄 Generating airfoil coordinates..."):
            x_coords, y_coords = generate_naca_coordinates(naca_code)
            scaled_input = scale_coordinates_for_prediction(x_coords, y_coords)
        
        # Airfoil preview
        st.markdown("### 🛩️ **Airfoil Geometry Preview**")
        
        fig_preview, ax_preview = plt.subplots(1, 1, figsize=(14, 5))
        ax_preview.plot(x_coords, y_coords, 'navy', linewidth=4, label=f'NACA {naca_code}')
        ax_preview.fill(x_coords, y_coords, alpha=0.3, color='lightblue')
        ax_preview.set_aspect('equal')
        ax_preview.grid(True, alpha=0.3)
        
        if selected_preset != "Custom (Manual Input)":
            title = f'NACA {naca_code} - {selected_preset}'
        else:
            title = f'NACA {naca_code} - Custom Configuration'
            
        ax_preview.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax_preview.set_xlabel('x/c', fontsize=12)
        ax_preview.set_ylabel('y/c', fontsize=12)
        ax_preview.set_xlim(-0.05, 1.05)
        
        # Add airfoil information
        max_thickness = max(y_coords) - min(y_coords)
        info_text = f'Max t/c: {max_thickness:.3f}\nCamber: {camber}%\nThickness: {thickness}%'
        
        if selected_preset != "Custom (Manual Input)":
            preset_info = aircraft_presets[selected_preset]
            info_text += f'\n\nApplication:\n{preset_info["description"]}'
        
        ax_preview.text(0.02, 0.98, info_text, 
                       transform=ax_preview.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.7", facecolor="white", alpha=0.9))
        
        st.pyplot(fig_preview)
        
        # Prediction section
        st.markdown("### 🧠 **Neural Network Prediction**")
        
        col_pred1, col_pred2 = st.columns([1, 2])
        
        with col_pred1:
            st.markdown("**Ready to predict Cp distribution?**")
            
            if selected_preset != "Custom (Manual Input)":
                st.markdown(f"Analyzing **{selected_preset}** airfoil configuration.")
            
            predict_button = st.button(
                "🚀 **Predict Cp Distribution**",
                type="primary",
                use_container_width=True
            )
            
        with col_pred2:
            if selected_preset != "Custom (Manual Input)":
                preset_details = aircraft_presets[selected_preset]['details']
                st.info(f"""
                **{selected_preset} - NACA {naca_code}**
                
                {preset_details}
                
                **Model Information:**
                - Zero angle of attack, subsonic flow
                - Inviscid assumptions
                - Instant prediction vs. hours of CFD
                """)
            else:
                st.info("""
                **Custom Airfoil Analysis**
                
                **Model Information:**
                - Input: 400 scaled airfoil coordinates
                - Output: 200 pressure coefficient values
                - Conditions: Zero angle of attack, subsonic flow
                - Prediction time: < 1 second
                """)
        
        # Prediction execution
        if predict_button:
            st.markdown("---")
            st.markdown(f"### 📊 **Prediction Results - {selected_preset if selected_preset != 'Custom (Manual Input)' else 'Custom Airfoil'}**")
            
            with st.spinner(f"🧠 Analyzing {selected_preset.lower() if selected_preset != 'Custom (Manual Input)' else 'custom'} airfoil aerodynamics..."):
                try:
                    # Make prediction
                    prediction = model.predict(scaled_input.reshape(1, -1), verbose=0)
                    cp_distribution = prediction[0]
                    
                    # Extract upper and lower surface Cp
                    cp_upper = cp_distribution[:100]
                    cp_lower = cp_distribution[100:]
                    x_cp = np.linspace(0, 1, 100)
                    
                    st.success("✅ Aerodynamic analysis completed successfully!")
                    
                    # Display prediction statistics
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    
                    with col_stats1:
                        st.metric("Min Cp (Peak Suction)", f"{cp_distribution.min():.3f}")
                    
                    with col_stats2:
                        st.metric("Max Cp (Max Pressure)", f"{cp_distribution.max():.3f}")
                        
                    with col_stats3:
                        st.metric("Cp Range", f"{cp_distribution.max() - cp_distribution.min():.3f}")
                        
                    with col_stats4:
                        st.metric("Upper Surface Min", f"{cp_upper.min():.3f}")
                    
                    # Comprehensive visualization
                    st.markdown("#### 📈 **Airfoil Analysis Results**")
                    
                    fig_results, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # 1. Airfoil geometry
                    ax1.plot(x_coords, y_coords, 'navy', linewidth=3, label='Airfoil Shape')
                    ax1.fill(x_coords, y_coords, alpha=0.2, color='lightblue')
                    ax1.set_aspect('equal')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_title(f'NACA {naca_code} - Geometry', fontweight='bold')
                    ax1.set_xlabel('x/c')
                    ax1.set_ylabel('y/c')
                    ax1.legend()
                    
                    # 2. Complete Cp distribution
                    ax2.plot(x_cp, cp_upper, 'red', linewidth=3, marker='o', markersize=4, 
                            label='Upper Surface', markerfacecolor='red')
                    ax2.plot(x_cp, cp_lower, 'blue', linewidth=3, marker='s', markersize=4,
                            label='Lower Surface', markerfacecolor='blue')
                    ax2.invert_yaxis()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title(f'Predicted Cp Distribution - NACA {naca_code}', fontweight='bold')
                    ax2.set_xlabel('x/c')
                    ax2.set_ylabel('Pressure Coefficient (Cp)')
                    ax2.legend()
                    
                    # Add peak suction annotation
                    min_cp_idx = np.argmin(cp_upper)
                    ax2.annotate(f'Peak Suction\nCp = {cp_upper[min_cp_idx]:.3f}', 
                                xy=(x_cp[min_cp_idx], cp_upper[min_cp_idx]),
                                xytext=(0.3, cp_upper[min_cp_idx] - 0.5),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                fontsize=10, ha='center',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                    
                    # 3. Upper surface Cp detail
                    ax3.plot(x_cp, cp_upper, 'red', linewidth=2, marker='o', markersize=3)
                    ax3.invert_yaxis()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_title('Upper Surface Cp (Detailed)', fontweight='bold')
                    ax3.set_xlabel('x/c')
                    ax3.set_ylabel('Cp')
                    ax3.fill_between(x_cp, cp_upper, alpha=0.3, color='red')
                    
                    # 4. Lower surface Cp detail
                    ax4.plot(x_cp, cp_lower, 'blue', linewidth=2, marker='s', markersize=3)
                    ax4.invert_yaxis()
                    ax4.grid(True, alpha=0.3)
                    ax4.set_title('Lower Surface Cp (Detailed)', fontweight='bold') 
                    ax4.set_xlabel('x/c')
                    ax4.set_ylabel('Cp')
                    ax4.fill_between(x_cp, cp_lower, alpha=0.3, color='blue')
                    
                    plt.tight_layout()
                    st.pyplot(fig_results)
                    
                    # Data export options
                    st.markdown("#### 💾 **Export Results**")
                    
                    results_data = pd.DataFrame({
                        'x_c': x_cp,
                        'Cp_upper': cp_upper,
                        'Cp_lower': cp_lower
                    })
                    
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        csv_data = results_data.to_csv(index=False)
                        st.download_button(
                            label="📄 Download Cp Data (CSV)",
                            data=csv_data,
                            file_name=f"NACA_{naca_code}_Cp_distribution.csv",
                            mime="text/csv"
                        )
                    
                    with col_export2:
                        summary_text = f"""NACA {naca_code} Analysis Report
Generated by Neural Network Model

Aircraft Configuration: {selected_preset}

Airfoil Parameters:
- Camber: {camber}%
- Camber Position: {position*10}%  
- Thickness: {thickness}%

Pressure Coefficient Results:
- Minimum Cp: {cp_distribution.min():.4f}
- Maximum Cp: {cp_distribution.max():.4f}
- Cp Range: {cp_distribution.max() - cp_distribution.min():.4f}
- Upper Surface Min Cp: {cp_upper.min():.4f}
- Lower Surface Max Cp: {cp_lower.max():.4f}
"""
                        
                        st.download_button(
                            label="📋 Download Analysis Report",
                            data=summary_text,
                            file_name=f"NACA_{naca_code}_analysis_report.txt",
                            mime="text/plain"
                        )
                    
                    # Show data table
                    with st.expander("📊 View Detailed Cp Data"):
                        st.dataframe(results_data, use_container_width=True)
                        
                        st.markdown("**Quick Statistics:**")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.write(f"Upper Surface Mean Cp: {cp_upper.mean():.3f}")
                        with col_stat2:
                            st.write(f"Lower Surface Mean Cp: {cp_lower.mean():.3f}")
                        with col_stat3:
                            st.write(f"Overall Mean Cp: {cp_distribution.mean():.3f}")
                    
                    # Enhanced results interpretation
                    with st.expander("🎓 **Aerodynamic Analysis & Interpretation**"):
                        st.markdown(f"""
                        ### NACA {naca_code} - {selected_preset} Analysis
                        
                        **Pressure Characteristics:**
                        - **Peak Suction**: {cp_distribution.min():.3f} (typically occurs near leading edge)
                        - **Pressure Range**: {cp_distribution.max() - cp_distribution.min():.3f} (indicates lift-generating potential)
                        - **Upper Surface Dominance**: {'Strong' if abs(cp_upper.min()) > abs(cp_lower.max()) else 'Moderate'} suction contribution
                        
                        **Engineering Implications:**
                        """)
                        
                        if selected_preset != "Custom (Manual Input)":
                            preset_analysis = aircraft_presets[selected_preset]
                            st.markdown(f"""
                            **{selected_preset} Context:**
                            - This pressure distribution is characteristic of {preset_analysis['description'].lower()}
                            - {preset_analysis['details']}
                            
                            **Design Trade-offs:**
                            - Higher suction peaks indicate better lift generation but may increase drag
                            - Pressure recovery characteristics affect stall behavior
                            - Overall distribution supports the intended {selected_preset.lower()} mission requirements
                            """)
                        else:
                            st.markdown("""
                            **Custom Configuration Analysis:**
                            - Your airfoil shows pressure characteristics typical of its parameter combination
                            - Consider how these results align with your intended application requirements
                            - Compare with preset aircraft configurations for context
                            """)
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.error("Please check your model file and try again.")
    
    except Exception as e:
        st.error(f"❌ Error generating airfoil: {str(e)}")
        st.error("Please check your NACA parameters and try again.")

# ============================================================================
# MULTIPLE AIRFOIL ANALYSIS (NEW FUNCTIONALITY)
# ============================================================================

def multiple_airfoil_analysis(model):
    """New multiple airfoil comparative analysis functionality"""
    
    st.markdown("## 🔄 **Multiple Airfoil Comparative Analysis**")
    st.markdown("Compare up to 4 different airfoils side-by-side to understand their aerodynamic characteristics.")
    st.markdown("---")
    
    # Get preset configurations
    aircraft_presets = get_aircraft_presets()
    
    # Number of airfoils selector
    st.sidebar.markdown("## ⚙️ Analysis Configuration")
    num_airfoils = st.sidebar.selectbox(
        "**Number of Airfoils to Compare**",
        [2, 3, 4],
        index=0,
        help="Select how many airfoils you want to analyze simultaneously"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛩️ Quick Presets")
    
    # Quick preset configurations for common comparisons
    preset_comparisons = {
        "Custom Configuration": "Configure each airfoil manually",
        "Commercial vs Military": ["Commercial Passenger Aircraft", "Military Fighter Aircraft"],
        "Thickness Comparison": ["Custom (Manual Input)", "Custom (Manual Input)", "Custom (Manual Input)"],
        "Aircraft Categories": ["Commercial Passenger Aircraft", "Business Jet", "General Aviation Training"],
        "Extreme Comparison": ["Military Fighter Aircraft", "Cargo Transport Aircraft", "Wind Turbine Blade"],
    }
    
    quick_preset = st.sidebar.selectbox(
        "**Quick Comparison Presets**",
        list(preset_comparisons.keys()),
        index=0,
        help="Select a predefined comparison or choose Custom for manual setup"
    )
    
    # Configuration for each airfoil
    airfoil_configs = {}
    
    # Create columns for airfoil configuration
    if num_airfoils == 2:
        col1, col2 = st.columns(2)
        containers = [col1, col2]
    elif num_airfoils == 3:
        col1, col2, col3 = st.columns(3)
        containers = [col1, col2, col3]
    else:  # 4 airfoils
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        containers = [col1, col2, col3, col4]
    
    # Configure each airfoil
    colors = ['red', 'blue', 'green', 'orange']
    line_styles = ['-', '--', '-.', ':']
    
    for i in range(num_airfoils):
        airfoil_key = f"airfoil_{i+1}"
        
        # Determine default preset based on quick preset selection
        if quick_preset == "Custom Configuration":
            default_preset = "Custom (Manual Input)"
        elif quick_preset in preset_comparisons and isinstance(preset_comparisons[quick_preset], list):
            if i < len(preset_comparisons[quick_preset]):
                default_preset = preset_comparisons[quick_preset][i]
            else:
                default_preset = "Custom (Manual Input)"
        else:
            default_preset = "Custom (Manual Input)"
        
        # Configure airfoil
        naca_code, selected_preset, camber, position, thickness = configure_airfoil(
            airfoil_key, containers[i], aircraft_presets, default_preset
        )
        
        # Store configuration
        airfoil_configs[airfoil_key] = {
            'naca_code': naca_code,
            'selected_preset': selected_preset,
            'camber': camber,
            'position': position,
            'thickness': thickness,
            'color': colors[i],
            'linestyle': line_styles[i]
        }
    
    st.markdown("---")
    
    # Analysis button
    col_analyze1, col_analyze2, col_analyze3 = st.columns([1, 2, 1])
    
    with col_analyze2:
        analyze_button = st.button(
            "🚀 **Analyze All Airfoils**",
            type="primary",
            use_container_width=True,
            help="Generate coordinates and predict Cp distributions for all configured airfoils"
        )
    
    # Perform analysis
    if analyze_button:
        st.markdown("---")
        st.markdown("## 📊 **Comparative Analysis Results**")
        
        # Generate coordinates and predictions for all airfoils
        airfoil_data = {}
        
        with st.spinner("🔄 Generating coordinates and making predictions for all airfoils..."):
            for airfoil_key, config in airfoil_configs.items():
                try:
                    # Generate coordinates
                    x_coords, y_coords = generate_naca_coordinates(config['naca_code'])
                    scaled_input = scale_coordinates_for_prediction(x_coords, y_coords)
                    
                    # Make prediction
                    prediction = model.predict(scaled_input.reshape(1, -1), verbose=0)
                    cp_distribution = prediction[0]
                    
                    # Store data
                    airfoil_data[airfoil_key] = {
                        'x_coords': x_coords,
                        'y_coords': y_coords,
                        'cp_distribution': cp_distribution,
                        'cp_upper': cp_distribution[:100],
                        'cp_lower': cp_distribution[100:],
                        'x_cp': np.linspace(0, 1, 100),
                        **config
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing {config['naca_code']}: {str(e)}")
                    return
        
        st.success("✅ All airfoils analyzed successfully!")
        
        # Summary comparison table
        st.markdown("### 📋 **Quick Comparison Summary**")
        
        summary_data = []
        for airfoil_key, data in airfoil_data.items():
            summary_data.append({
                'Airfoil': f"Airfoil {airfoil_key.split('_')[1]}",
                'NACA Code': data['naca_code'],
                'Aircraft Type': data['selected_preset'].split()[0] if data['selected_preset'] != "Custom (Manual Input)" else "Custom",
                'Camber (%)': data['camber'],
                'Thickness (%)': data['thickness'],
                'Min Cp': f"{data['cp_distribution'].min():.3f}",
                'Max Cp': f"{data['cp_distribution'].max():.3f}",
                'Cp Range': f"{data['cp_distribution'].max() - data['cp_distribution'].min():.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Visualization section
        st.markdown("### 📈 **Comparative Visualizations**")
        
        # Create comprehensive comparison plots
        fig_comparison, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. All airfoil geometries overlaid
        ax1.set_title('Airfoil Geometry Comparison', fontweight='bold', fontsize=14)
        for airfoil_key, data in airfoil_data.items():
            label = f"NACA {data['naca_code']} ({data['selected_preset'].split()[0] if data['selected_preset'] != 'Custom (Manual Input)' else 'Custom'})"
            ax1.plot(data['x_coords'], data['y_coords'], 
                    color=data['color'], linewidth=3, 
                    linestyle=data['linestyle'], label=label)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x/c')
        ax1.set_ylabel('y/c')
        ax1.legend()
        ax1.set_xlim(-0.05, 1.05)
        
        # 2. Upper surface Cp comparison
        ax2.set_title('Upper Surface Cp Comparison', fontweight='bold', fontsize=14)
        for airfoil_key, data in airfoil_data.items():
            label = f"NACA {data['naca_code']}"
            ax2.plot(data['x_cp'], data['cp_upper'], 
                    color=data['color'], linewidth=3,
                    linestyle=data['linestyle'], marker='o', markersize=3,
                    label=label)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x/c')
        ax2.set_ylabel('Cp')
        ax2.legend()
        
        # 3. Lower surface Cp comparison
        ax3.set_title('Lower Surface Cp Comparison', fontweight='bold', fontsize=14)
        for airfoil_key, data in airfoil_data.items():
            label = f"NACA {data['naca_code']}"
            ax3.plot(data['x_cp'], data['cp_lower'], 
                    color=data['color'], linewidth=3,
                    linestyle=data['linestyle'], marker='s', markersize=3,
                    label=label)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('x/c')
        ax3.set_ylabel('Cp')
        ax3.legend()
        
        # 4. Complete Cp comparison
        ax4.set_title('Complete Cp Distribution Comparison', fontweight='bold', fontsize=14)
        for airfoil_key, data in airfoil_data.items():
            label_upper = f"NACA {data['naca_code']} (Upper)"
            label_lower = f"NACA {data['naca_code']} (Lower)"
            ax4.plot(data['x_cp'], data['cp_upper'], 
                    color=data['color'], linewidth=2,
                    linestyle='-', alpha=0.8, label=label_upper)
            ax4.plot(data['x_cp'], data['cp_lower'], 
                    color=data['color'], linewidth=2,
                    linestyle='--', alpha=0.8, label=label_lower)
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('x/c')
        ax4.set_ylabel('Cp')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig_comparison)
        
        # Detailed comparison metrics
        st.markdown("### 📊 **Detailed Performance Comparison**")
        
        # Performance metrics comparison
        metrics_data = []
        for airfoil_key, data in airfoil_data.items():
            metrics_data.append({
                'Airfoil': f"NACA {data['naca_code']}",
                'Aircraft Type': data['selected_preset'].split()[0] if data['selected_preset'] != "Custom (Manual Input)" else "Custom",
                'Peak Suction (Min Cp)': data['cp_distribution'].min(),
                'Max Pressure (Max Cp)': data['cp_distribution'].max(),
                'Pressure Range': data['cp_distribution'].max() - data['cp_distribution'].min(),
                'Upper Surface Min Cp': data['cp_upper'].min(),
                'Lower Surface Max Cp': data['cp_lower'].max(),
                'Upper Surface Mean Cp': data['cp_upper'].mean(),
                'Lower Surface Mean Cp': data['cp_lower'].mean(),
                'Overall Mean Cp': data['cp_distribution'].mean()
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display metrics with color coding
        st.dataframe(
            metrics_df.style.format({
                'Peak Suction (Min Cp)': '{:.4f}',
                'Max Pressure (Max Cp)': '{:.4f}',
                'Pressure Range': '{:.4f}',
                'Upper Surface Min Cp': '{:.4f}',
                'Lower Surface Max Cp': '{:.4f}',
                'Upper Surface Mean Cp': '{:.4f}',
                'Lower Surface Mean Cp': '{:.4f}',
                'Overall Mean Cp': '{:.4f}'
            }).background_gradient(subset=['Peak Suction (Min Cp)', 'Pressure Range'], cmap='RdYlBu'),
            use_container_width=True,
            hide_index=True
        )
        
        # Export comparison data
        st.markdown("### 💾 **Export Comparison Data**")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Export summary data
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Summary (CSV)",
                data=summary_csv,
                file_name=f"airfoil_comparison_summary_{num_airfoils}airfoils.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            # Export detailed metrics
            metrics_csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Detailed Metrics (CSV)",
                data=metrics_csv,
                file_name=f"airfoil_detailed_metrics_{num_airfoils}airfoils.csv",
                mime="text/csv"
            )
        
        with col_exp3:
            # Export complete Cp data for all airfoils
            complete_cp_data = pd.DataFrame()
            complete_cp_data['x_c'] = airfoil_data[list(airfoil_data.keys())[0]]['x_cp']
            
            for airfoil_key, data in airfoil_data.items():
                naca_code = data['naca_code']
                complete_cp_data[f'{naca_code}_upper'] = data['cp_upper']
                complete_cp_data[f'{naca_code}_lower'] = data['cp_lower']
            
            complete_csv = complete_cp_data.to_csv(index=False)
            st.download_button(
                label="🔄 Download All Cp Data (CSV)",
                data=complete_csv,
                file_name=f"complete_cp_data_{num_airfoils}airfoils.csv",
                mime="text/csv"
            )
        
        # Comparative analysis insights
        with st.expander("🎓 **Comparative Analysis Insights**"):
            st.markdown("### 🔍 **Key Findings**")
            
            # Find extremes
            min_cp_airfoil = min(airfoil_data.items(), key=lambda x: x[1]['cp_distribution'].min())
            max_range_airfoil = max(airfoil_data.items(), key=lambda x: x[1]['cp_distribution'].max() - x[1]['cp_distribution'].min())
            thickest_airfoil = max(airfoil_data.items(), key=lambda x: x[1]['thickness'])
            
            st.markdown(f"""
            **Performance Highlights:**
            
            🔥 **Highest Suction**: NACA {min_cp_airfoil[1]['naca_code']} with Cp = {min_cp_airfoil[1]['cp_distribution'].min():.3f}
            - Aircraft Type: {min_cp_airfoil[1]['selected_preset']}
            - This indicates the strongest lift-generating capability
            
            📈 **Largest Pressure Range**: NACA {max_range_airfoil[1]['naca_code']} with range = {max_range_airfoil[1]['cp_distribution'].max() - max_range_airfoil[1]['cp_distribution'].min():.3f}
            - Aircraft Type: {max_range_airfoil[1]['selected_preset']}
            - This suggests the highest lift potential
            
            🏗️ **Thickest Section**: NACA {thickest_airfoil[1]['naca_code']} with {thickest_airfoil[1]['thickness']}% thickness
            - Aircraft Type: {thickest_airfoil[1]['selected_preset']}
            - Provides structural strength and internal volume
            
            **Engineering Trade-offs:**
            - Higher camber typically increases lift but may increase drag
            - Thicker sections provide structure but increase drag at high speeds
            - Symmetric airfoils (0% camber) work equally well upright and inverted
            - Different applications require different compromises between lift, drag, and structural requirements
            """)
            
            # Application-specific insights
            st.markdown("### 🛩️ **Application Context**")
            for airfoil_key, data in airfoil_data.items():
                if data['selected_preset'] != "Custom (Manual Input)":
                    preset_info = aircraft_presets[data['selected_preset']]
                    st.markdown(f"""
                    **NACA {data['naca_code']} - {data['selected_preset']}:**
                    - {preset_info['description']}
                    - Peak Suction: {data['cp_distribution'].min():.3f}
                    - Design Philosophy: {preset_info['details']}
                    """)

# ============================================================================
# MAIN APPLICATION WITH PAGE SELECTION
# ============================================================================

def main():
    """Enhanced application with single and multiple airfoil analysis"""
    
    # App header
    st.title("✈️ Neural Network Airfoil Cp Predictor")
    st.markdown("**Interactive NACA 4-Digit Airfoil Analysis with Deep Learning**")
    st.markdown("---")
    
    # Load model
    with st.spinner("🤖 Loading neural network model..."):
        model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("Please ensure your model file is in the correct location.")
        st.stop()
    
    # ========================================================================
    # PAGE SELECTION
    # ========================================================================
    
    st.sidebar.markdown("# 🎯 Analysis Mode")
    
    analysis_mode = st.sidebar.radio(
        "**Select Analysis Type**",
        ["Single Airfoil Analysis", "Multiple Airfoil Comparison"],
        index=0,
        help="Choose between analyzing a single airfoil or comparing multiple airfoils"
    )
    
    st.sidebar.markdown("---")
    
    # Route to appropriate analysis function
    if analysis_mode == "Single Airfoil Analysis":
        single_airfoil_analysis(model)
    else:
        multiple_airfoil_analysis(model)
    
    # ========================================================================
    # ENHANCED FOOTER WITH COMPREHENSIVE INFO
    # ========================================================================
    
    st.markdown("---")
    
    with st.expander("📚 **Aircraft Preset Database Information**"):
        st.markdown("### 🛩️ Available Aircraft Configurations")
        aircraft_presets = get_aircraft_presets()
        
        for name, config in aircraft_presets.items():
            if name != "Custom (Manual Input)":
                st.markdown(f"""
                **{name}** - NACA {config['naca']}
                - *{config['description']}*
                - Parameters: {config['camber']}% camber, {config['position']*10}% position, {config['thickness']}% thickness
                """)
        
        st.markdown("""
        ### 🎯 Preset Selection Benefits
        - **Educational**: Learn about different aircraft applications
        - **Benchmarking**: Compare your custom designs with proven configurations
        - **Quick Start**: Instantly load realistic airfoil parameters
        - **Context**: Understand real-world engineering trade-offs
        """)
    
    with st.expander("ℹ️ **Model Information & Limitations**"):
        st.markdown("""
        ### 🤖 Neural Network Model Details
        
        **Architecture:**
        - Input: 400 scaled airfoil coordinates
        - Hidden Layers: 512 → 256 → 128 neurons
        - Output: 200 pressure coefficient values
        - Total Parameters: ~250,000
        
        **Training Data:**
        - 1000+ NACA 4-digit airfoils
        - Systematic parameter coverage
        - Custom panel method CFD
        
        **Performance:**
        - High accuracy on test data
        - Prediction Time: <1 second
        - Reliable for preliminary analysis
        
        **Limitations:**
        - NACA 4-digit airfoils only
        - Zero angle of attack only
        - Inviscid flow assumptions
        - Subsonic conditions only
        
        **Citation:**
        Neural Network Airfoil Pressure Prediction Model
        Developed using TensorFlow/Keras
        """)
    
    with st.expander("🔄 **Multiple Airfoil Analysis Features**"):
        st.markdown("""
        ### 🆕 New Multiple Airfoil Comparison Mode
        
        **Capabilities:**
        - Compare 2-4 airfoils simultaneously
        - Side-by-side geometry visualization
        - Comparative Cp distribution analysis
        - Performance metrics comparison
        - Quick preset comparisons
        
        **Use Cases:**
        - **Design Optimization**: Test parameter variations
        - **Trade-off Analysis**: Compare different aircraft applications
        - **Educational**: Understand airfoil design principles
        - **Research**: Systematic airfoil studies
        
        **Export Features:**
        - Summary comparison tables
        - Detailed performance metrics
        - Complete Cp datasets for all airfoils
        - Professional analysis reports
        
        **Quick Presets Available:**
        - Commercial vs Military comparison
        - Thickness variation studies  
        - Aircraft category comparisons
        - Extreme configuration analysis
        """)
    
    st.markdown("---")
    st.markdown("*Enhanced with ❤️ using Python, TensorFlow, and Streamlit*")
    st.markdown("*Now featuring Multiple Airfoil Comparative Analysis! 🚀*")

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()