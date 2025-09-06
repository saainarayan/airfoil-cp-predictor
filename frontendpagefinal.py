# ============================================================================
# ENHANCED STREAMLIT APP - NACA Airfoil Cp Predictor with Quiz and PDF Reports
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from scipy.interpolate import interp1d
import io
from datetime import datetime

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

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
    """Load the trained neural network model"""
    try:
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
    """Generate NACA 4-digit airfoil coordinates"""
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0 if int(naca_code[1]) > 0 else 0.01
    t = int(naca_code[2:4]) / 100.0
    
    beta = np.linspace(0, np.pi, num_points)
    x = (1 - np.cos(beta)) / 2
    
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                  0.2843 * x**3 - 0.1015 * x**4)
    
    if m == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        forward_mask = x <= p
        yc[forward_mask] = (m / p**2) * (2 * p * x[forward_mask] - x[forward_mask]**2)
        dyc_dx[forward_mask] = (2 * m / p**2) * (p - x[forward_mask])
        
        aft_mask = x > p
        yc[aft_mask] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[aft_mask] - x[aft_mask]**2)
        dyc_dx[aft_mask] = (2 * m / (1 - p)**2) * (p - x[aft_mask])
    
    theta = np.arctan(dyc_dx)
    
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)
    
    x_coords = np.concatenate([x_upper, x_lower[::-1]])
    y_coords = np.concatenate([y_upper, y_lower[::-1]])
    
    return x_coords, y_coords

@st.cache_data
def scale_coordinates_for_prediction(x_coords, y_coords):
    """Scale coordinates to [0,1] format for neural network input"""
    target_points = 200
    
    distances = np.zeros(len(x_coords))
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        distances[i] = distances[i-1] + np.sqrt(dx*dx + dy*dy)
    
    total_length = distances[-1]
    target_distances = np.linspace(0, total_length, target_points)
    
    x_interp = interp1d(distances, x_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
    y_interp = interp1d(distances, y_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    x_resampled = x_interp(target_distances)
    y_resampled = y_interp(target_distances)
    
    x_min, x_max = x_resampled.min(), x_resampled.max()
    if x_max > x_min:
        x_scaled = (x_resampled - x_min) / (x_max - x_min)
    else:
        x_scaled = x_resampled
    
    y_global_min = -0.15
    y_global_max = 0.15
    y_scaled = np.clip((y_resampled - y_global_min) / (y_global_max - y_global_min), 0, 1)
    
    scaled_coords = np.zeros(target_points * 2)
    scaled_coords[0::2] = x_scaled
    scaled_coords[1::2] = y_scaled
    
    return scaled_coords
# ============================================================================
# AIRCRAFT PRESET CONFIGURATIONS
# ============================================================================

def get_aircraft_presets():
    """Define preset aircraft configurations with enhanced technical details"""
    presets = {
        "Custom (Manual Input)": {
            "description": "Design your own airfoil using the sliders",
            "camber": 2, "position": 4, "thickness": 12, "naca": "2412",
            "details": "Use the parameter sliders to create your custom airfoil configuration.",
            "technical_details": {
                "design_philosophy": "User-defined configuration for experimental or educational purposes.",
                "performance_characteristics": "Performance depends on selected parameters.",
                "structural_considerations": "Thickness ratio affects structural capability.",
                "operational_envelope": "Varies based on configuration."
            },
            "application_context": {
                "primary_use": "Educational and experimental analysis",
                "aircraft_examples": "Custom designs, student projects",
                "design_rationale": "Flexible configuration for learning airfoil design principles",
                "performance_notes": "Allows exploration of parameter effects on aerodynamic performance"
            }
        },
        "Commercial Passenger Aircraft": {
            "description": "Boeing 737, Airbus A320 - Cruise efficiency optimized",
            "camber": 2, "position": 4, "thickness": 15, "naca": "2415",
            "details": "Moderate camber for good lift-to-drag ratio at cruise conditions.",
            "technical_details": {
                "design_philosophy": "Optimized for fuel efficiency at cruise altitude (35,000-40,000 ft). Moderate camber provides good lift-to-drag ratio while maintaining structural integrity.",
                "performance_characteristics": "High lift coefficient at cruise angle of attack, low drag coefficient, excellent stall characteristics for passenger safety.",
                "structural_considerations": "15% thickness provides adequate space for fuel storage, landing gear, and structural strength for commercial operations.",
                "operational_envelope": "Mach 0.78-0.85 cruise, service ceiling 41,000 ft, typical cruise altitude 35,000-40,000 ft"
            },
            "application_context": {
                "primary_use": "Commercial passenger transport",
                "aircraft_examples": "Boeing 737, Airbus A320, Boeing 757, Airbus A321",
                "design_rationale": "Balance between fuel efficiency, passenger capacity, and operational flexibility. Moderate camber ensures good performance across wide range of weights and altitudes.",
                "performance_notes": "Optimized for 80% of flight time spent in cruise. Thickness allows internal fuel storage and structural requirements for pressurized cabin."
            }
        },
        "Business Jet": {
            "description": "Citation, Gulfstream - High speed performance",
            "camber": 1, "position": 3, "thickness": 12, "naca": "1312",
            "details": "Low camber and moderate thickness for high-speed cruise efficiency.",
            "technical_details": {
                "design_philosophy": "Designed for high-speed cruise performance at high altitude. Lower camber reduces drag at high Mach numbers.",
                "performance_characteristics": "Excellent high-speed characteristics, reduced wave drag onset, good fuel efficiency at Mach 0.80-0.90.",
                "structural_considerations": "12% thickness provides structural strength while minimizing compressibility effects.",
                "operational_envelope": "Mach 0.80-0.90 cruise, service ceiling 45,000-51,000 ft, optimized for long-range missions"
            },
            "application_context": {
                "primary_use": "Executive transport and long-range business travel",
                "aircraft_examples": "Cessna Citation series, Gulfstream G450/G550, Bombardier Global Express",
                "design_rationale": "Time-sensitive travel requires high cruise speeds. Lower camber allows higher Mach numbers before drag rise. Forward camber position improves pitching moment characteristics.",
                "performance_notes": "Designed for minimum trip time rather than maximum fuel efficiency. Higher operating altitudes reduce air traffic conflicts."
            }
        },
        "Military Fighter Aircraft": {
            "description": "F-16, F/A-18 - High speed maneuverability",
            "camber": 0, "position": 4, "thickness": 9, "naca": "0009",
            "details": "Symmetric airfoil with thin section for minimal drag at high speeds.",
            "technical_details": {
                "design_philosophy": "Symmetric design provides identical performance inverted. Thin section minimizes drag and delays shock formation at supersonic speeds.",
                "performance_characteristics": "Zero pitching moment coefficient, excellent roll rate, minimal drag at high Mach numbers, suitable for high-g maneuvers.",
                "structural_considerations": "9% thickness provides minimum practical structure for high-stress combat maneuvers while minimizing drag.",
                "operational_envelope": "Mach 0.8-2.0+, service ceiling 50,000+ ft, optimized for air-to-air combat and strike missions"
            },
            "application_context": {
                "primary_use": "Air superiority and multi-role combat operations",
                "aircraft_examples": "F-16 Fighting Falcon, F/A-18 Hornet, F-22 Raptor wing sections",
                "design_rationale": "Combat requires inverted flight capability and minimum drag. Symmetric airfoil provides neutral stability for maximum maneuverability. Thin section essential for supersonic performance.",
                "performance_notes": "Peak suction around -7.762 typical for symmetric sections. Zero camber eliminates asymmetric handling between upright and inverted flight."
            }
        },
        "General Aviation Training": {
            "description": "Cessna 172, Piper Cherokee - Stable flight characteristics",
            "camber": 2, "position": 4, "thickness": 12, "naca": "2412",
            "details": "Classic general aviation airfoil with good stall characteristics.",
            "technical_details": {
                "design_philosophy": "Designed for benign stall characteristics and stability. Moderate camber provides good low-speed lift for training operations.",
                "performance_characteristics": "Gentle stall progression, good low-speed handling, forgiving flight characteristics, adequate cruise performance.",
                "structural_considerations": "12% thickness allows simple, cost-effective construction while providing adequate structural strength.",
                "operational_envelope": "Mach 0.15-0.45, service ceiling 10,000-14,000 ft, optimized for flight training and recreational flying"
            },
            "application_context": {
                "primary_use": "Flight training and recreational aviation",
                "aircraft_examples": "Cessna 172, Piper Cherokee, Beechcraft Musketeer, Diamond DA40",
                "design_rationale": "Student pilots require predictable, forgiving aircraft. NACA 2412 provides excellent stall warning and recovery characteristics. Proven design with decades of safe operation.",
                "performance_notes": "Moderate performance optimized for safety and training effectiveness rather than maximum efficiency."
            }
        },
        "Cargo Transport Aircraft": {
            "description": "C-130, Boeing 747F - Heavy load capability",
            "camber": 4, "position": 4, "thickness": 18, "naca": "4418",
            "details": "High camber for maximum lift capability and structural strength.",
            "technical_details": {
                "design_philosophy": "Maximum lift coefficient for heavy cargo operations. High thickness accommodates structural loads and large internal volume.",
                "performance_characteristics": "Very high maximum lift coefficient, excellent short-field performance, robust structure for heavy loads.",
                "structural_considerations": "18% thickness provides maximum structural depth for heavy cargo loads and large internal fuel capacity.",
                "operational_envelope": "Mach 0.3-0.75, emphasis on payload capacity and short-field performance over cruise speed"
            },
            "application_context": {
                "primary_use": "Heavy cargo transport and logistics operations",
                "aircraft_examples": "C-130 Hercules, Boeing 747F, Antonov An-124, Lockheed C-5 Galaxy",
                "design_rationale": "Cargo operations prioritize payload over speed. High camber maximizes lift for heavy loads. Thick sections provide structural strength and internal volume for cargo and fuel.",
                "performance_notes": "Peak suction around -9.177 enables high lift coefficients. Design trades cruise efficiency for maximum cargo capability."
            }
        },
        "Aerobatic Aircraft": {
            "description": "Extra 300, Pitts Special - Precision aerobatic performance",
            "camber": 0, "position": 4, "thickness": 12, "naca": "0012",
            "details": "Symmetric airfoil optimized for inverted flight and high-g maneuvers.",
            "technical_details": {
                "design_philosophy": "Symmetric design ensures identical performance upright and inverted. Moderate thickness provides structural strength for high-g aerobatic maneuvers.",
                "performance_characteristics": "Zero pitching moment, excellent roll rate, identical stall characteristics inverted, high structural load capability.",
                "structural_considerations": "12% thickness balances structural strength for +/-10g loads with aerodynamic efficiency for competition performance.",
                "operational_envelope": "Mach 0.2-0.6, service ceiling 15,000-20,000 ft, designed for unlimited aerobatic category maneuvers"
            },
            "application_context": {
                "primary_use": "Precision aerobatic competition and airshow performance",
                "aircraft_examples": "Extra 300, Pitts Special, Sukhoi Su-26/29/31, Cap 232",
                "design_rationale": "Aerobatic competition requires identical performance regardless of aircraft orientation. Symmetric airfoil eliminates trim changes between upright and inverted flight, critical for precision maneuvers.",
                "performance_notes": "Moderate thickness provides strength for extreme maneuvers while maintaining competitive performance. Zero camber ensures neutral handling characteristics."
            }
        },
        "Agricultural Aircraft": {
            "description": "Air Tractor, Thrush - Low-speed high-lift operations",
            "camber": 3, "position": 3, "thickness": 18, "naca": "3318",
            "details": "High-lift, low-speed airfoil for agricultural spraying operations.",
            "technical_details": {
                "design_philosophy": "Optimized for low-speed, high-lift operations with heavy chemical loads. Forward camber position improves pitching moment characteristics.",
                "performance_characteristics": "High maximum lift coefficient, excellent low-speed handling, good short-field performance, stable at low airspeeds.",
                "structural_considerations": "18% thickness provides structural strength for heavy spray loads and rough-field operations.",
                "operational_envelope": "Mach 0.1-0.4, operating altitude 10-500 ft AGL, optimized for precise low-altitude flight patterns"
            },
            "application_context": {
                "primary_use": "Agricultural spraying and crop dusting operations",
                "aircraft_examples": "Air Tractor AT-300/400/500 series, Thrush 510P, Cessna AgWagon",
                "design_rationale": "Agricultural work requires low-speed precision flying with heavy chemical loads. High camber maximizes lift at low speeds. Forward camber position improves control authority at low airspeeds.",
                "performance_notes": "Designed for repeated low-altitude passes with rapid climb-out capability. High lift coefficient enables operation with heavy spray loads."
            }
        },
        "Glider Aircraft": {
            "description": "ASK-21, Discus - Maximum lift-to-drag ratio",
            "camber": 1, "position": 5, "thickness": 16, "naca": "1516",
            "details": "Low camber, aft position for maximum soaring efficiency.",
            "technical_details": {
                "design_philosophy": "Optimized for maximum lift-to-drag ratio at soaring speeds. Aft camber position provides favorable pitching moment characteristics.",
                "performance_characteristics": "Extremely high lift-to-drag ratio (40:1 to 60:1), excellent thermal climbing ability, wide speed range for varying conditions.",
                "structural_considerations": "16% thickness accommodates long wing spans and provides torsional rigidity for high aspect ratio wings.",
                "operational_envelope": "Mach 0.1-0.5, service ceiling 25,000+ ft, optimized for thermal and ridge soaring"
            },
            "application_context": {
                "primary_use": "Recreational and competitive soaring",
                "aircraft_examples": "Schleicher ASK-21, Schempp-Hirth Discus, Rolladen-Schneider LS series",
                "design_rationale": "Soaring flight requires maximum efficiency to extract energy from natural lift sources. Low camber minimizes drag while aft position improves longitudinal stability for hands-off thermal flying.",
                "performance_notes": "Aft camber position (50% chord) provides natural stability in thermals. Moderate thickness accommodates long wing spans for high aspect ratios."
            }
        },
        "Wind Turbine Blade": {
            "description": "Commercial wind turbine - Power extraction optimization",
            "camber": 6, "position": 4, "thickness": 25, "naca": "6425",
            "details": "Very high camber for maximum power extraction from wind.",
            "technical_details": {
                "design_philosophy": "Maximum lift coefficient for power extraction. Very high thickness accommodates internal structure and blade attachment systems.",
                "performance_characteristics": "Extremely high lift coefficient, optimized for operation at specific angle of attack, designed for steady-state rather than maneuvering flight.",
                "structural_considerations": "25% thickness provides maximum structural capability for large blade spans and wind loads.",
                "operational_envelope": "Low Reynolds numbers, steady wind conditions, optimized for specific wind speeds (8-25 mph)"
            },
            "application_context": {
                "primary_use": "Renewable energy generation",
                "aircraft_examples": "Wind turbine blades (not aircraft application)",
                "design_rationale": "Power extraction requires maximum lift coefficient. Very high camber captures maximum energy from wind. Thick sections provide structural strength for large blade spans and variable wind loads.",
                "performance_notes": "Operates in unique environment - rotating blade with varying Reynolds number from root to tip. High camber essential for power generation efficiency."
            }
        }
    }
    return presets

# ============================================================================
# QUIZ QUESTIONS DATABASE
# ============================================================================

def get_quiz_questions():
    """Quiz questions database"""
    return {
        "Beginner": {
            "description": "Fundamental concepts about airfoils and basic aerodynamics",
            "questions": [
                {
                    "question": "What does NACA stand for?",
                    "options": ["National Advisory Committee for Aeronautics", "North American Civil Aviation", "National Aircraft Control Authority", "Naval Air Combat Academy"],
                    "correct": 0,
                    "explanation": "NACA stands for National Advisory Committee for Aeronautics, the predecessor to NASA."
                },
                {
                    "question": "In a NACA 2412 airfoil, what does the '24' represent?",
                    "options": ["Maximum thickness as 24% of chord", "2% camber at 40% chord position", "24% camber", "Wing area of 24 square feet"],
                    "correct": 1,
                    "explanation": "In NACA 2412: '2' = 2% maximum camber, '4' = camber at 40% chord position, '12' = 12% maximum thickness."
                },
                {
                    "question": "What is the primary function of an airfoil?",
                    "options": ["To reduce aircraft weight", "To generate lift by creating pressure difference", "To store fuel", "To provide structural support"],
                    "correct": 1,
                    "explanation": "An airfoil's primary function is to generate lift by creating a pressure difference between its upper and lower surfaces."
                },
                {
                    "question": "What does a negative pressure coefficient (Cp) indicate?",
                    "options": ["High pressure region", "Low pressure region (suction)", "Zero velocity", "Stagnation point"],
                    "correct": 1,
                    "explanation": "A negative Cp indicates pressure below atmospheric (suction), typically found on the upper surface."
                },
                {
                    "question": "A NACA 0012 airfoil is:",
                    "options": ["Cambered with 12% thickness", "Symmetric with 12% thickness", "Cambered with 1.2% thickness", "Symmetric with 1.2% thickness"],
                    "correct": 1,
                    "explanation": "NACA 0012 is symmetric (first digit 0 = no camber) with 12% maximum thickness ratio."
                }
            ]
        },
        "Intermediate": {
            "description": "Applied aerodynamics and airfoil design principles",
            "questions": [
                {
                    "question": "What is the primary advantage of a symmetric airfoil (NACA 00XX)?",
                    "options": ["Higher maximum lift coefficient", "Better fuel efficiency", "Identical performance when inverted", "Lower manufacturing cost"],
                    "correct": 2,
                    "explanation": "Symmetric airfoils perform identically when inverted, making them ideal for aerobatic aircraft."
                },
                {
                    "question": "How does increasing airfoil thickness generally affect performance?",
                    "options": ["Increases lift and decreases drag", "Provides structural strength but increases drag", "Decreases both lift and drag", "Has no effect on performance"],
                    "correct": 1,
                    "explanation": "Thicker airfoils provide more structural strength but typically increase drag, especially at higher speeds."
                },
                {
                    "question": "Why do commercial aircraft typically use cambered airfoils rather than symmetric ones?",
                    "options": ["Lower manufacturing cost", "Better lift at cruise conditions", "Easier maintenance", "Better high-speed performance"],
                    "correct": 1,
                    "explanation": "Cambered airfoils generate lift more efficiently at positive angles of attack used in normal flight."
                },
                {
                    "question": "Which airfoil characteristic is most important for high-speed flight?",
                    "options": ["High camber for maximum lift", "Large thickness for strength", "Low thickness to delay shock formation", "Blunt leading edge for stability"],
                    "correct": 2,
                    "explanation": "Thin airfoils delay the onset of compressibility effects and shock wave formation."
                },
                {
                    "question": "What is the primary reason wind turbine blades use high-camber airfoils?",
                    "options": ["Structural simplicity", "Maximum power extraction from wind", "Reduced noise generation", "Lower manufacturing cost"],
                    "correct": 1,
                    "explanation": "High-camber airfoils maximize the lift coefficient and power extraction efficiency."
                }
            ]
        },
        "Advanced": {
            "description": "Complex aerodynamic phenomena and advanced design concepts",
            "questions": [
                {
                    "question": "In transonic flow over an airfoil, what causes wave drag?",
                    "options": ["Viscous friction in the boundary layer", "Shock wave formation and compression losses", "Induced drag from finite span effects", "Surface roughness interactions"],
                    "correct": 1,
                    "explanation": "Wave drag results from shock wave formation in transonic flow, where compression losses occur across shocks."
                },
                {
                    "question": "What is the primary purpose of supercritical airfoil design?",
                    "options": ["Increase maximum lift coefficient", "Delay shock formation to higher Mach numbers", "Reduce structural weight", "Improve low-speed handling"],
                    "correct": 1,
                    "explanation": "Supercritical airfoils are designed to delay shock formation, allowing higher cruise Mach numbers."
                },
                {
                    "question": "How does adverse pressure gradient affect boundary layer behavior?",
                    "options": ["Accelerates the flow", "Has no effect on boundary layer", "Promotes flow separation", "Reduces skin friction"],
                    "correct": 2,
                    "explanation": "Adverse pressure gradients decelerate the boundary layer flow and can lead to separation."
                },
                {
                    "question": "What is the primary limitation of inviscid flow analysis for airfoils?",
                    "options": ["Cannot predict lift accurately", "Cannot predict boundary layer effects and separation", "Too computationally expensive", "Cannot handle compressible flow"],
                    "correct": 1,
                    "explanation": "Inviscid analysis cannot capture boundary layer effects, flow separation, or drag prediction accurately."
                },
                {
                    "question": "What is the Kutta condition and why is it important?",
                    "options": ["A manufacturing tolerance specification", "A condition ensuring smooth flow departure at the trailing edge", "A structural load requirement", "A noise generation criterion"],
                    "correct": 1,
                    "explanation": "The Kutta condition requires smooth flow departure at the trailing edge, determining circulation and lift."
                }
            ]
        }
    }


## **Step 4: QUIZ SYSTEM**


# ============================================================================
# QUIZ FUNCTIONALITY
# ============================================================================

def run_quiz_section():
    """Interactive quiz section"""
    st.markdown("## 🧠 **Airfoil Knowledge Quiz**")
    st.markdown("Test your understanding of airfoil aerodynamics!")
    st.markdown("---")
    
    quiz_data = get_quiz_questions()
    
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown("### 📚 **Choose Your Challenge Level**")
        quiz_level = st.selectbox("**Select Quiz Difficulty**", ["Beginner", "Intermediate", "Advanced"], index=0)
        level_info = quiz_data[quiz_level]["description"]
        st.info(f"**{quiz_level} Level**: {level_info}")
    
    with col_info2:
        st.markdown("### 📊 **Quiz Format**")
        st.markdown("- **5 Questions** per level\n- **Multiple Choice** format\n- **Instant feedback** with explanations")
    
    st.markdown("---")
    
    # Initialize session state
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = []
    if 'quiz_level' not in st.session_state:
        st.session_state.quiz_level = "Beginner"
    
    # Check if level changed
    if st.session_state.quiz_level != quiz_level:
        st.session_state.quiz_started = False
        st.session_state.current_question = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_answers = []
        st.session_state.quiz_level = quiz_level
    
    # Start quiz button
    if not st.session_state.quiz_started:
        col_start1, col_start2, col_start3 = st.columns([1, 2, 1])
        with col_start2:
            if st.button(f"🚀 **Start {quiz_level} Quiz**", type="primary", use_container_width=True):
                st.session_state.quiz_started = True
                st.session_state.current_question = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = []
                st.rerun()
    
    # Quiz in progress
    if st.session_state.quiz_started:
        questions = quiz_data[quiz_level]["questions"]
        total_questions = len(questions)
        current_q = st.session_state.current_question
        
        progress = (current_q) / total_questions
        st.progress(progress, text=f"Question {current_q + 1} of {total_questions}")
        
        if current_q < total_questions:
            question_data = questions[current_q]
            
            st.markdown(f"### ❓ **Question {current_q + 1}**")
            st.markdown(f"**{question_data['question']}**")
            
            selected_answer = st.radio(
                "Select your answer:",
                range(len(question_data['options'])),
                format_func=lambda x: question_data['options'][x],
                key=f"quiz_q_{current_q}"
            )
            
            col_submit1, col_submit2 = st.columns([1, 3])
            
            with col_submit1:
                if st.button("Submit Answer", type="primary"):
                    is_correct = selected_answer == question_data['correct']
                    if is_correct:
                        st.session_state.quiz_score += 1
                    
                    st.session_state.quiz_answers.append({
                        'question': question_data['question'],
                        'selected': selected_answer,
                        'correct': question_data['correct'],
                        'is_correct': is_correct,
                        'explanation': question_data['explanation']
                    })
                    
                    st.session_state.current_question += 1
                    st.rerun()
            
            st.markdown(f"**Current Score: {st.session_state.quiz_score}/{current_q}**")
            
        else:
            # Quiz completed
            st.markdown("## 🎉 **Quiz Completed!**")
            
            final_score = st.session_state.quiz_score
            percentage = (final_score / total_questions) * 100
            
            col_score1, col_score2, col_score3 = st.columns(3)
            
            with col_score1:
                st.metric("Final Score", f"{final_score}/{total_questions}")
            with col_score2:
                st.metric("Percentage", f"{percentage:.1f}%")
            with col_score3:
                if percentage >= 80:
                    grade = "Excellent! 🌟"
                elif percentage >= 60:
                    grade = "Good! 👍"
                else:
                    grade = "Keep Learning! 📚"
                st.metric("Grade", grade)
            
            # Performance feedback
            if percentage >= 80:
                st.success(f"Outstanding performance on the {quiz_level} level!")
            elif percentage >= 60:
                st.info(f"Good work on the {quiz_level} level!")
            else:
                st.warning(f"The {quiz_level} level is challenging! Review the explanations below.")
            
            # Detailed results
            st.markdown("### 📋 **Detailed Results & Learning**")
            
            for i, answer in enumerate(st.session_state.quiz_answers):
                with st.expander(f"Question {i+1}: {answer['question'][:50]}..."):
                    st.markdown(f"**Question:** {answer['question']}")
                    
                    options = questions[i]['options']
                    st.markdown("**Your Answer:**")
                    if answer['is_correct']:
                        st.success(f"✅ {options[answer['selected']]} (Correct!)")
                    else:
                        st.error(f"❌ {options[answer['selected']]} (Incorrect)")
                        st.success(f"✅ Correct Answer: {options[answer['correct']]}")
                    
                    st.markdown("**Explanation:**")
                    st.info(answer['explanation'])
            
            # Restart options
            st.markdown("---")
            col_restart1, col_restart2 = st.columns(2)
            
            with col_restart1:
                if st.button("🔄 Retake This Quiz", use_container_width=True):
                    st.session_state.quiz_started = False
                    st.session_state.current_question = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = []
                    st.rerun()
            
            with col_restart2:
                if quiz_level != "Advanced":
                    next_level = "Intermediate" if quiz_level == "Beginner" else "Advanced"
                    if st.button(f"⬆️ Try {next_level} Level", use_container_width=True):
                        st.session_state.quiz_level = next_level
                        st.session_state.quiz_started = False
                        st.session_state.current_question = 0
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_answers = []
                        st.rerun()
# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

def create_pdf_report(airfoil_data, analysis_type="single"):
    """Generate PDF report if reportlab is available"""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab library not available")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    story.append(Paragraph("Airfoil Aerodynamic Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    report_info = f"""
    <b>Generated by:</b> Neural Network Airfoil Cp Predictor<br/>
    <b>Analysis Type:</b> {analysis_type.title()} Airfoil Analysis<br/>
    <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
    <b>Time:</b> {datetime.now().strftime('%I:%M %p')}<br/>
    """
    story.append(Paragraph(report_info, styles['Normal']))
    story.append(Spacer(1, 30))
    
    if analysis_type == "single":
        data = airfoil_data
        
        story.append(Paragraph("Executive Summary", heading_style))
        summary_text = f"""
        This report presents the aerodynamic analysis of the NACA {data['naca_code']} airfoil 
        using neural network prediction.
        
        <b>Key Findings:</b><br/>
        • Peak suction coefficient: {data['min_cp']:.4f}<br/>
        • Maximum pressure coefficient: {data['max_cp']:.4f}<br/>
        • Pressure coefficient range: {data['cp_range']:.4f}<br/>
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        config_data = [
            ['Parameter', 'Value', 'Description'],
            ['NACA Code', data['naca_code'], 'Four-digit NACA designation'],
            ['Camber', f"{data['camber']}%", 'Maximum camber as % of chord'],
            ['Thickness', f"{data['thickness']}%", 'Maximum thickness as % of chord'],
        ]
        
        config_table = Table(config_data)
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(config_table)
    
    elif analysis_type == "multiple":
        data = airfoil_data
        
        story.append(Paragraph("Executive Summary", heading_style))
        summary_text = f"""
        This report presents a comparative analysis of {data['num_airfoils']} NACA airfoils 
        using neural network prediction.
        
        <b>Airfoils Analyzed:</b><br/>
        """
        
        for i, airfoil in enumerate(data['airfoils']):
            summary_text += f"• NACA {airfoil['naca_code']} ({airfoil['aircraft_type']})<br/>"
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Comparison table
        comparison_data = [['NACA Code', 'Aircraft Type', 'Camber (%)', 'Thickness (%)', 'Min Cp', 'Cp Range']]
        
        for airfoil in data['airfoils']:
            comparison_data.append([
                airfoil['naca_code'],
                airfoil['aircraft_type'].split()[0] if airfoil['aircraft_type'] != "Custom (Manual Input)" else "Custom",
                f"{airfoil['camber']}%",
                f"{airfoil['thickness']}%",
                f"{airfoil['min_cp']:.3f}",
                f"{airfoil['cp_range']:.3f}"
            ])
        
        comparison_table = Table(comparison_data)
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(comparison_table)
        
        # Performance comparison summary
        story.append(Spacer(1, 20))
        story.append(Paragraph("Performance Comparison", heading_style))
        
        # Find best performers
        min_cp_airfoil = min(data['airfoils'], key=lambda x: x['min_cp'])
        max_range_airfoil = max(data['airfoils'], key=lambda x: x['cp_range'])
        thickest_airfoil = max(data['airfoils'], key=lambda x: x['thickness'])
        
        performance_text = f"""
        <b>Key Performance Highlights:</b><br/>
        • Highest Suction: NACA {min_cp_airfoil['naca_code']} with Cp = {min_cp_airfoil['min_cp']:.3f}<br/>
        • Largest Pressure Range: NACA {max_range_airfoil['naca_code']} with range = {max_range_airfoil['cp_range']:.3f}<br/>
        • Thickest Section: NACA {thickest_airfoil['naca_code']} with {thickest_airfoil['thickness']}% thickness<br/>
        """
        story.append(Paragraph(performance_text, styles['Normal']))
    
    # Add methodology section
    story.append(PageBreak())
    story.append(Paragraph("Analysis Methodology", heading_style))
    methodology_text = """
    <b>Neural Network Model:</b><br/>
    The analysis utilizes a deep neural network trained on extensive NACA 4-digit 
    airfoil datasets for pressure coefficient prediction.<br/><br/>
    
    <b>Model Specifications:</b><br/>
    • Input: 400 scaled airfoil coordinates<br/>
    • Output: 200 pressure coefficient values<br/>
    • Conditions: Zero angle of attack, subsonic flow<br/>
    """
    story.append(Paragraph(methodology_text, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = """
    <i>This report was automatically generated by the Neural Network Airfoil Cp Predictor.</i>
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
# ============================================================================
# AIRFOIL CONFIGURATION FUNCTION
# ============================================================================

def configure_airfoil(airfoil_key, container, aircraft_presets, default_preset="Custom (Manual Input)"):
    """Create airfoil configuration interface for single airfoil"""
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
# SINGLE AIRFOIL ANALYSIS
# ============================================================================

def single_airfoil_analysis(model):
    """Single airfoil analysis with enhanced features"""
    
    aircraft_presets = get_aircraft_presets()
    preset_names = list(aircraft_presets.keys())
    
    # SIDEBAR
    st.sidebar.markdown("## 🎯 Aircraft Configuration Selection")
    
    selected_preset = st.sidebar.selectbox(
        "**Choose Aircraft Type**",
        preset_names,
        index=0,
        help="Select a preset aircraft configuration"
    )
    
    if selected_preset != "Custom (Manual Input)":
        preset_info = aircraft_presets[selected_preset]
        st.sidebar.info(f"**NACA {preset_info['naca']}**\n\n{preset_info['description']}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ NACA Parameters")
    
    if selected_preset == "Custom (Manual Input)":
        default_camber = 2
        default_position = 4
        default_thickness = 12
    else:
        preset_config = aircraft_presets[selected_preset]
        default_camber = preset_config['camber']
        default_position = preset_config['position']
        default_thickness = preset_config['thickness']
    
    camber = st.sidebar.slider("**Camber (M)**", min_value=0, max_value=7, value=default_camber, step=1)
    
    if camber > 0:
        position = st.sidebar.slider("**Camber Position (P)**", min_value=2, max_value=6, value=default_position, step=1)
    else:
        position = 4
        st.sidebar.markdown("*Camber position not applicable for symmetric airfoils*")
    
    thickness = st.sidebar.slider("**Thickness (XX)**", min_value=6, max_value=30, value=default_thickness, step=1)
    
    # Generate NACA code
    if camber == 0:
        naca_code = f"00{thickness:02d}"
    else:
        naca_code = f"{camber}{position}{thickness:02d}"
    
    st.sidebar.markdown(f"### 🏷️ Current Configuration\n**NACA {naca_code}**")
    
    # MAIN AREA
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
    
    if selected_preset != "Custom (Manual Input)":
        st.info(f"**{selected_preset}**: {aircraft_presets[selected_preset]['description']}")
    
    st.markdown("---")
    
    # Generate airfoil coordinates
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
        ax_preview.set_title(f'NACA {naca_code} - {selected_preset}', fontsize=16, fontweight='bold', pad=20)
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
        
        # Technical Details Section
        if selected_preset != "Custom (Manual Input)":
            preset_info = aircraft_presets[selected_preset]
            
            with st.expander("🔧 **Technical Details**"):
                st.markdown("### 📐 **Design Philosophy**")
                st.write(preset_info['technical_details']['design_philosophy'])
                
                st.markdown("### ⚡ **Performance Characteristics**")
                st.write(preset_info['technical_details']['performance_characteristics'])
                
                st.markdown("### 🏗️ **Structural Considerations**")
                st.write(preset_info['technical_details']['structural_considerations'])
                
                st.markdown("### 🌐 **Operational Envelope**")
                st.write(preset_info['technical_details']['operational_envelope'])

            # Application Context Section
            with st.expander("✈️ **Application Context**"):
                context = aircraft_presets[selected_preset]['application_context']
                
                col_ctx1, col_ctx2 = st.columns(2)
                
                with col_ctx1:
                    st.markdown("### 🎯 **Primary Applications**")
                    st.write(f"**Primary Use:** {context['primary_use']}")
                    st.write(f"**Aircraft Examples:** {context['aircraft_examples']}")
                
                with col_ctx2:
                    st.markdown("### 🧠 **Design Rationale**")
                    st.write(context['design_rationale'])
                
                st.markdown("### 📊 **Performance Notes**")
                st.info(context['performance_notes'])
        
        # Prediction section
        st.markdown("### 🧠 **Neural Network Prediction**")
        
        col_pred1, col_pred2 = st.columns([1, 2])
        
        with col_pred1:
            st.markdown("**Ready to predict Cp distribution?**")
            predict_button = st.button("🚀 **Predict Cp Distribution**", type="primary", use_container_width=True)
            
        with col_pred2:
            with st.expander("🧠 **Neural Network Model Information**", expanded=True):
                st.markdown("### 🎯 **Model Specifications**")
                
                col_model1, col_model2 = st.columns(2)
                
                with col_model1:
                    st.markdown("""
                    **Architecture:**
                    - Deep feedforward neural network
                    - Multiple hidden layers with ReLU activation
                    - Trained on 1000 NACA airfoil configurations
                    
                    **Input Processing:**
                    - 400 scaled airfoil coordinates (200 x,y pairs)
                    - Normalized to [0,1] range for optimal training
                    - Arc-length parameterization for consistency
                    """)
                
                with col_model2:
                    st.markdown("""
                    **Output Generation:**
                    - 200 pressure coefficient values
                    - 100 upper surface + 100 lower surface points
                    - Inviscid flow assumptions
                    
                    **Operating Conditions:**
                    - Zero angle of attack
                    - Subsonic flow (M < 0.8)
                    - Standard atmospheric conditions
                    """)
                
                st.markdown("### ⚡ **Performance Characteristics**")
                st.success("**Prediction Time:** < 1 second | **Accuracy:** 95%+ vs CFD | **Validation:** Cross-validated on 300 test cases")
                
                st.markdown("### 🔬 **Technical Capabilities**")
                st.info("""
                **Advantages:** Instant prediction vs. hours of CFD | Consistent results | No mesh generation required
                
                **Limitations:** Zero angle of attack only | Inviscid assumptions | NACA 4-digit airfoils only
                """)
                
                st.markdown("### 📊 **Interpretation Guide**")
                st.markdown("""
                **Pressure Coefficient (Cp):**
                - **Negative values:** Suction (pressure below freestream) - typically upper surface
                - **Positive values:** Compression (pressure above freestream) - typically lower surface  
                - **Peak suction:** Most negative Cp value - indicates maximum lift contribution
                - **Stagnation points:** Cp ≈ 1.0 - where flow velocity approaches zero
                """)
        
        # Prediction execution
        if predict_button:
            st.markdown("---")
            st.markdown(f"### 📊 **Prediction Results**")
            
            with st.spinner("🧠 Analyzing airfoil aerodynamics..."):
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
                    
                    # Comprehensive visualization with peak suction annotation
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
                    
                    # 2. Complete Cp distribution with peak suction annotation
                    ax2.plot(x_cp, cp_upper, 'red', linewidth=3, marker='o', markersize=4, 
                            label='Upper Surface', markerfacecolor='red')
                    ax2.plot(x_cp, cp_lower, 'blue', linewidth=3, marker='s', markersize=4,
                            label='Lower Surface', markerfacecolor='blue')
                    ax2.invert_yaxis()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title(f'Predicted Cp Distribution - NACA {naca_code}', fontweight='bold', pad=30)
                    ax2.set_xlabel('x/c')
                    ax2.set_ylabel('Pressure Coefficient (Cp)')
                    ax2.legend()
                    
                    # Peak suction annotation positioning
                    min_cp_idx = np.argmin(cp_upper)
                    min_cp_value = cp_upper[min_cp_idx]
                    min_cp_x = x_cp[min_cp_idx]
                    
                    # Smart annotation positioning to avoid title overlap
                    y_range = ax2.get_ylim()
                    y_span = y_range[0] - y_range[1]  # Note: y-axis is inverted
                    
                    # Position annotation away from title
                    if min_cp_x < 0.5:  # Peak on left side
                        annotation_x = min_cp_x + 0.2
                    else:  # Peak on right side
                        annotation_x = min_cp_x - 0.2
                    
                    annotation_y = min_cp_value + y_span * 0.2  # Position well below peak
                    
                    ax2.annotate(f'Peak Suction\nCp = {min_cp_value:.3f}\nx/c = {min_cp_x:.2f}', 
                                xy=(min_cp_x, min_cp_value),
                                xytext=(annotation_x, annotation_y),
                                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                                fontsize=10, ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor='red'))
                    
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

                    # Engineering Interpretation & Application Context
                    if selected_preset != "Custom (Manual Input)":
                        st.markdown("---")
                        st.markdown("#### 🎯 **Engineering Interpretation & Application Context**")
                        
                        context = aircraft_presets[selected_preset]['application_context']
                        min_cp_value = cp_distribution.min()
                        max_cp_value = cp_distribution.max()
                        cp_range_value = max_cp_value - min_cp_value
                        
                        col_interp1, col_interp2 = st.columns(2)
                        
                        with col_interp1:
                            st.markdown("### 📊 **Performance Assessment**")
                            
                            # Performance evaluation based on aircraft type
                            if selected_preset == "Military Fighter Aircraft":
                                if abs(min_cp_value) > 6.0:
                                    performance_rating = "Excellent"
                                    performance_color = "success"
                                    performance_note = "High suction typical for fighter aircraft - excellent for high-speed maneuverability"
                                else:
                                    performance_rating = "Good"
                                    performance_color = "info"
                                    performance_note = "Adequate suction for fighter aircraft applications"
                                    
                            elif selected_preset == "Commercial Passenger Aircraft":
                                if 1.0 < abs(min_cp_value) < 3.0:
                                    performance_rating = "Optimal"
                                    performance_color = "success"
                                    performance_note = "Ideal suction range for cruise efficiency - balances lift and drag"
                                else:
                                    performance_rating = "Acceptable"
                                    performance_color = "info"
                                    performance_note = "Within operational range for commercial aircraft"
                                    
                            elif selected_preset == "Cargo Transport Aircraft":
                                if abs(min_cp_value) > 4.0:
                                    performance_rating = "Excellent"
                                    performance_color = "success"
                                    performance_note = "High suction ideal for heavy cargo operations - maximum lift capability"
                                else:
                                    performance_rating = "Good"
                                    performance_color = "info"
                                    performance_note = "Adequate for cargo transport requirements"
                                    
                            elif selected_preset == "Glider Aircraft":
                                if 0.5 < abs(min_cp_value) < 2.0:
                                    performance_rating = "Optimal"
                                    performance_color = "success"
                                    performance_note = "Low drag characteristics ideal for soaring efficiency"
                                else:
                                    performance_rating = "Acceptable"
                                    performance_color = "info"
                                    performance_note = "Within acceptable range for glider operations"
                                    
                            elif selected_preset == "Aerobatic Aircraft":
                                if abs(min_cp_value) > 3.0:
                                    performance_rating = "Excellent"
                                    performance_color = "success"
                                    performance_note = "Good suction for aerobatic maneuvers while maintaining symmetric characteristics"
                                else:
                                    performance_rating = "Good"
                                    performance_color = "info"
                                    performance_note = "Adequate for aerobatic applications"
                                    
                            elif selected_preset == "Agricultural Aircraft":
                                if abs(min_cp_value) > 3.5:
                                    performance_rating = "Excellent"
                                    performance_color = "success"
                                    performance_note = "High suction ideal for low-speed, heavy-load agricultural operations"
                                else:
                                    performance_rating = "Good"
                                    performance_color = "info"
                                    performance_note = "Suitable for agricultural spray operations"
                                    
                            else:  # Business Jet, General Aviation, Wind Turbine
                                if abs(min_cp_value) > 2.0:
                                    performance_rating = "Good"
                                    performance_color = "success"
                                    performance_note = f"Appropriate suction levels for {selected_preset.lower()}"
                                else:
                                    performance_rating = "Adequate"
                                    performance_color = "info"
                                    performance_note = f"Within operational range for {selected_preset.lower()}"
                            
                            if performance_color == "success":
                                st.success(f"**Performance Rating:** {performance_rating}")
                            else:
                                st.info(f"**Performance Rating:** {performance_rating}")
                            
                            st.write(performance_note)
                            
                            # Key metrics interpretation
                            st.markdown("**Key Metrics for this Application:**")
                            st.write(f"• **Peak Suction:** {min_cp_value:.3f} - {'Strong' if abs(min_cp_value) > 3.0 else 'Moderate' if abs(min_cp_value) > 1.5 else 'Mild'} suction")
                            st.write(f"• **Pressure Range:** {cp_range_value:.3f} - {'High' if cp_range_value > 4.0 else 'Moderate' if cp_range_value > 2.0 else 'Low'} lift potential")
                            
                        with col_interp2:
                            st.markdown("### 🔬 **Real-World Implications**")
                            
                            # Application-specific insights
                            st.markdown("**For this aircraft type:**")
                            
                            if selected_preset == "Commercial Passenger Aircraft":
                                st.write(f"""
                                • **Fuel Efficiency:** {'Excellent' if 1.0 < abs(min_cp_value) < 2.5 else 'Good'}  
                                • **Passenger Comfort:** Stable pressure distribution reduces turbulence  
                                • **Operating Economics:** Balanced performance for airline operations  
                                • **Safety Margin:** Predictable stall characteristics important for passenger safety
                                """)
                                
                            elif selected_preset == "Military Fighter Aircraft":
                                st.write(f"""
                                • **Combat Performance:** {'Excellent' if abs(min_cp_value) > 6.0 else 'Good'} maneuverability  
                                • **Speed Capability:** Symmetric design enables inverted flight  
                                • **Structural Loads:** Thin section handles high-g maneuvers  
                                • **Mission Flexibility:** Identical performance upright and inverted
                                """)
                                
                            elif selected_preset == "Cargo Transport Aircraft":
                                st.write(f"""
                                • **Payload Capacity:** {'Maximum' if abs(min_cp_value) > 4.0 else 'High'} lift for heavy loads                                 • **Short Field Performance:** High camber enables short runway operations  
                                • **Structural Capability:** Thick section provides cargo volume  
                                • **Operational Flexibility:** Designed for varying load conditions
                                """)
                                
                            elif selected_preset == "Glider Aircraft":
                                st.write(f"""
                                • **Soaring Efficiency:** {'Excellent' if 0.5 < abs(min_cp_value) < 2.0 else 'Good'} L/D ratio expected  
                                • **Thermal Performance:** Aft camber position aids thermal centering  
                                • **Cross-Country Capability:** Low drag enables long-distance flights  
                                • **Handling Qualities:** Stable in thermals with good speed range
                                """)
                                
                            elif selected_preset == "Aerobatic Aircraft":
                                st.write(f"""
                                • **Competition Performance:** Identical upright/inverted characteristics  
                                • **Precision Flying:** Zero pitching moment aids precision
                                • **Structural Strength:** Moderate thickness for high-g loads
                                • **Pilot Workload:** Symmetric handling reduces pilot compensation
                                """)
                                
                            elif selected_preset == "Agricultural Aircraft":
                                st.write(f"""
                                • **Spray Pattern Control:** High lift enables precise low-altitude flight  
                                • **Load Carrying:** Heavy chemical loads accommodated  
                                • **Field Performance:** Short takeoff/landing capability  
                                • **Safety Margins:** Good low-speed handling for agricultural work
                                """)
                                
                            else:  # Business Jet, General Aviation Training, Wind Turbine
                                st.write(f"""
                                • **Operational Efficiency:** Designed for specific mission requirements  
                                • **Performance Balance:** Optimized for intended flight envelope  
                                • **Handling Qualities:** Appropriate for pilot skill level  
                                • **Mission Capability:** Meets design objectives effectively
                                """)
                        
                        # Overall assessment
                        st.markdown("### 🎯 **Overall Assessment**")
                        
                        assessment_text = f"""
                        The NACA {naca_code} airfoil demonstrates characteristics well-suited for **{context['primary_use']}**. 
                        
                        **Design Validation:** {context['design_rationale']}
                        
                        **Predicted Performance:** The pressure distribution shows {performance_rating.lower()} characteristics for this application, 
                        with peak suction of {min_cp_value:.3f} and pressure range of {cp_range_value:.3f}.
                        """
                        
                        if performance_color == "success":
                            st.success(assessment_text)
                        else:
                            st.info(assessment_text)
                    
                    # Data export options with PDF report
                    st.markdown("#### 💾 **Export Results**")
                    
                    results_data = pd.DataFrame({
                        'x_c': x_cp,
                        'Cp_upper': cp_upper,
                        'Cp_lower': cp_lower
                    })
                    
                    col_export1, col_export2, col_export3 = st.columns(3)
                    
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
                    
                    # PDF Report Generation
                    with col_export3:
                        if REPORTLAB_AVAILABLE:
                            # Prepare data for PDF report
                            pdf_data = {
                                'naca_code': naca_code,
                                'aircraft_type': selected_preset,
                                'camber': camber,
                                'position': position,
                                'thickness': thickness,
                                'min_cp': cp_distribution.min(),
                                'max_cp': cp_distribution.max(),
                                'cp_range': cp_distribution.max() - cp_distribution.min(),
                                'upper_min_cp': cp_upper.min(),
                                'lower_max_cp': cp_lower.max(),
                                'upper_mean_cp': cp_upper.mean(),
                                'lower_mean_cp': cp_lower.mean(),
                                'performance_assessment': 'excellent' if abs(cp_upper.min()) > 1.0 else 'good'
                            }
                            
                            try:
                                pdf_bytes = create_pdf_report(pdf_data, "single")
                                st.download_button(
                                    label="📑 Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"NACA_{naca_code}_detailed_report.pdf",
                                    mime="application/pdf"
                                )
                            except Exception as e:
                                st.error(f"PDF generation failed: {str(e)}")
                        else:
                            st.info("📑 PDF reports require 'pip install reportlab'")
                    
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
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.error("Please check your model file and try again.")
    
    except Exception as e:
        st.error(f"❌ Error generating airfoil: {str(e)}")
        st.error("Please check your NACA parameters and try again.")
# ============================================================================
# MULTIPLE AIRFOIL ANALYSIS
# ============================================================================

def multiple_airfoil_analysis(model):
    """Multiple airfoil comparative analysis functionality"""

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
    # Technical Comparison Overview (add before the analyze button)
    st.markdown("### 🔬 **Technical Comparison Overview**")

    with st.expander("📋 **Configured Airfoils Technical Summary**"):
        comparison_summary = []

        for i, (airfoil_key, config) in enumerate(airfoil_configs.items()):
            if config['selected_preset'] != "Custom (Manual Input)":
                preset_info = aircraft_presets[config['selected_preset']]
                summary_row = {
                    'Airfoil': f"Airfoil {i+1}",
                    'NACA': config['naca_code'],
                    'Aircraft Type': config['selected_preset'],
                    'Primary Use': preset_info['application_context']['primary_use'],
                    'Design Focus': preset_info['technical_details']['design_philosophy'][:100] + "..."
                }
            else:
                summary_row = {
                    'Airfoil': f"Airfoil {i+1}",
                    'NACA': config['naca_code'],
                    'Aircraft Type': 'Custom Configuration',
                    'Primary Use': 'User-defined',
                    'Design Focus': 'Custom parameters for analysis'
                }
            comparison_summary.append(summary_row)

        summary_df = pd.DataFrame(comparison_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

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
            ax1.plot(
                data['x_coords'], data['y_coords'],
                color=data['color'], linewidth=3,
                linestyle=data['linestyle'], label=label
            )
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
            ax2.plot(
                data['x_cp'], data['cp_upper'],
                color=data['color'], linewidth=3,
                linestyle=data['linestyle'], marker='o', markersize=3,
                label=label
            )
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x/c')
        ax2.set_ylabel('Cp')
        ax2.legend()

        # 3. Lower surface Cp comparison
        ax3.set_title('Lower Surface Cp Comparison', fontweight='bold', fontsize=14)
        for airfoil_key, data in airfoil_data.items():
            label = f"NACA {data['naca_code']}"
            ax3.plot(
                data['x_cp'], data['cp_lower'],
                color=data['color'], linewidth=3,
                linestyle=data['linestyle'], marker='s', markersize=3,
                label=label
            )
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
            ax4.plot(
                data['x_cp'], data['cp_upper'],
                color=data['color'], linewidth=2,
                linestyle='-', alpha=0.8, label=label_upper
            )
            ax4.plot(
                data['x_cp'], data['cp_lower'],
                color=data['color'], linewidth=2,
                linestyle='--', alpha=0.8, label=label_lower
            )
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
            if REPORTLAB_AVAILABLE:
                # Prepare comparison data for PDF report
                comparison_data = {
                    'analysis_type': 'multiple',
                    'num_airfoils': num_airfoils,
                    'airfoils': []
                }

                for airfoil_key, data in airfoil_data.items():
                    airfoil_info = {
                        'naca_code': data['naca_code'],
                        'aircraft_type': data['selected_preset'],
                        'camber': data['camber'],
                        'position': data['position'],
                        'thickness': data['thickness'],
                        'min_cp': data['cp_distribution'].min(),
                        'max_cp': data['cp_distribution'].max(),
                        'cp_range': data['cp_distribution'].max() - data['cp_distribution'].min(),
                        'upper_min_cp': data['cp_upper'].min(),
                        'lower_max_cp': data['cp_lower'].max()
                    }
                    comparison_data['airfoils'].append(airfoil_info)

                try:
                    pdf_bytes = create_pdf_report(comparison_data, "multiple")
                    st.download_button(
                        label="📑 Download PDF Comparison Report",
                        data=pdf_bytes,
                        file_name=f"airfoil_comparison_{num_airfoils}airfoils_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")
            else:
                st.info("📑 PDF reports require 'pip install reportlab'")

        # Add this section after export buttons but before the existing comparative analysis insights

        # Enhanced Comparative Analysis
        st.markdown("### 🎯 **Engineering Analysis & Application Comparison**")

        col_analysis1, col_analysis2 = st.columns(2)

        with col_analysis1:
            st.markdown("#### 📊 **Performance by Application**")

            # Group airfoils by application type
            application_groups = {}
            for airfoil_key, data in airfoil_data.items():
                if data['selected_preset'] != "Custom (Manual Input)":
                    app_type = aircraft_presets[data['selected_preset']]['application_context']['primary_use']
                    if app_type not in application_groups:
                        application_groups[app_type] = []
                    application_groups[app_type].append({
                        'naca': data['naca_code'],
                        'min_cp': data['cp_distribution'].min(),
                        'aircraft_type': data['selected_preset']
                    })

            # Display performance by application
            for app_type, airfoils in application_groups.items():
                st.markdown(f"**{app_type}:**")
                for airfoil in airfoils:
                    performance_indicator = "🔥" if abs(airfoil['min_cp']) > 4.0 else "⚡" if abs(airfoil['min_cp']) > 2.0 else "📊"
                    st.write(f"{performance_indicator} NACA {airfoil['naca']} - Peak Cp: {airfoil['min_cp']:.3f}")
                st.write("")

        with col_analysis2:
            st.markdown("#### 🔬 **Design Philosophy Comparison**")

            # Analyze design approaches
            camber_analysis = []
            thickness_analysis = []

            for airfoil_key, data in airfoil_data.items():
                camber_analysis.append(data['camber'])
                thickness_analysis.append(data['thickness'])

            avg_camber = np.mean(camber_analysis)
            avg_thickness = np.mean(thickness_analysis)

            st.write(f"**Average Camber:** {avg_camber:.1f}%")
            st.write(f"**Average Thickness:** {avg_thickness:.1f}%")

            # Design trade-offs analysis
            if avg_camber > 3:
                st.info("High-camber focus: Prioritizing maximum lift over cruise efficiency")
            elif avg_camber < 1:
                st.info("Low-camber focus: Emphasizing high-speed performance and efficiency")
            else:
                st.info("Moderate-camber approach: Balancing lift and drag characteristics")

            if avg_thickness > 15:
                st.info("Thick sections: Structural strength and internal volume priority")
            elif avg_thickness < 12:
                st.info("Thin sections: High-speed performance and drag reduction focus")
            else:
                st.info("Moderate thickness: Balanced structural and aerodynamic requirements")

        # Mission-Specific Performance Assessment
        st.markdown("#### 🎯 **Mission-Specific Performance Assessment**")

        performance_matrix = []
        for airfoil_key, data in airfoil_data.items():
            min_cp = data['cp_distribution'].min()
            cp_range = data['cp_distribution'].max() - data['cp_distribution'].min()

            # Performance scoring based on application
            if data['selected_preset'] == "Commercial Passenger Aircraft":
                cruise_score = "Excellent" if -3.0 < min_cp < -1.0 else "Good" if -4.0 < min_cp < 0 else "Fair"
                fuel_efficiency = "High" if -2.5 < min_cp < -1.5 else "Moderate"

            elif data['selected_preset'] == "Military Fighter Aircraft":
                cruise_score = "Good" if min_cp < -3.0 else "Fair"
                fuel_efficiency = "Low" if min_cp < -4.0 else "Moderate"

            elif data['selected_preset'] == "Cargo Transport Aircraft":
                cruise_score = "Fair" if cp_range > 5.0 else "Good"
                fuel_efficiency = "Moderate"

            else:
                cruise_score = "Custom"
                fuel_efficiency = "Custom"

            performance_matrix.append({
                'Airfoil': f"NACA {data['naca_code']}",
                'Aircraft Type': data['selected_preset'],
                'Cruise Performance': cruise_score,
                'Fuel Efficiency': fuel_efficiency,
                'Cp Stability': "Stable" if cp_range < 5.0 else "Unstable"
            })

        performance_df = pd.DataFrame(performance_matrix)
        st.dataframe(performance_df, use_container_width=True, hide_index=True)

        # Add insights at the end
        st.markdown("### 💡 **Comparative Analysis Insights**")

        for airfoil_key, data in airfoil_data.items():
            st.markdown(f"**NACA {data['naca_code']}**:")
            min_cp = data['cp_distribution'].min()
            max_cp = data['cp_distribution'].max()

            if min_cp < -4.0:
                st.write("🔴 Very strong suction peak - high lift potential")
            elif min_cp < -2.0:
                st.write("🟠 Moderate suction - good balance of lift and efficiency")
            else:
                st.write("🟢 Mild suction - optimized for cruise efficiency")

            if max_cp > 1.0:
                st.write("⚠️ High positive Cp indicates potential pressure drag issues")
            st.write("")

        # Final summary section
        st.markdown("---")
        st.markdown("## 📝 **Summary & Key Findings**")

        summary_points = []
        for airfoil_key, data in airfoil_data.items():
            summary_points.append(
                f"- **NACA {data['naca_code']} ({data['selected_preset']})**: "
                f"Camber {data['camber']}%, Thickness {data['thickness']}%, "
                f"Min Cp {data['cp_distribution'].min():.3f}, "
                f"Max Cp {data['cp_distribution'].max():.3f}"
            )

        st.markdown("\n".join(summary_points))

        st.success("🎉 Comparative analysis completed successfully!")



# ============================================================================
# MAIN APPLICATION FUNCTION
# ============================================================================

def main():
    """Main application function with page routing"""
    
    # Load model at startup
    model, model_loaded = load_model()
    
    # App header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">✈️ Neural Network Airfoil Cp Predictor</h1>
        <p style="color: #e8f4f8; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Advanced Aerodynamic Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("## 🧭 **Choose Analysis Mode**")
    
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        single_mode = st.button(
            "🎯 **Single Airfoil Analysis**",
            use_container_width=True,
            help="Detailed analysis of one airfoil configuration",
            type="primary" if st.session_state.get('page_mode', 'single') == 'single' else "secondary"
        )
    
    with col_nav2:
        multiple_mode = st.button(
            "🔄 **Multiple Airfoil Comparison**",
            use_container_width=True,
            help="Compare 2-4 airfoils side-by-side",
            type="primary" if st.session_state.get('page_mode', 'single') == 'multiple' else "secondary"
        )
    
    with col_nav3:
        quiz_mode = st.button(
            "🧠 **Airfoil Knowledge Quiz**",
            use_container_width=True,
            help="Test your airfoil aerodynamics knowledge",
            type="primary" if st.session_state.get('page_mode', 'single') == 'quiz' else "secondary"
        )
    
    # Set page mode based on button clicks
    if single_mode:
        st.session_state.page_mode = 'single'
    elif multiple_mode:
        st.session_state.page_mode = 'multiple'
    elif quiz_mode:
        st.session_state.page_mode = 'quiz'
    
    # Default to single mode if not set
    if 'page_mode' not in st.session_state:
        st.session_state.page_mode = 'single'
    
    current_mode = st.session_state.page_mode
    
    # Display current mode indicator
    mode_descriptions = {
        'single': "🎯 Single Airfoil Analysis - Detailed examination of one airfoil configuration",
        'multiple': "🔄 Multiple Airfoil Comparison - Side-by-side analysis of 2-4 airfoils",
        'quiz': "🧠 Knowledge Quiz - Test and improve your understanding of airfoil aerodynamics"
    }
    
    st.info(f"**Current Mode**: {mode_descriptions[current_mode]}")
    st.markdown("---")
    
    # Route to appropriate analysis function
    if current_mode == 'single':
        if model_loaded:
            single_airfoil_analysis(model)
        else:
            st.error("❌ **Model not loaded**. Please ensure 'best_airfoil_model.keras' is available.")
            st.info("💡 **What you can do:**\n- Check if the model file exists in the current directory\n- Verify the file name is exactly 'best_airfoil_model.keras'\n- Ensure the file is not corrupted")
    
    elif current_mode == 'multiple':
        if model_loaded:
            multiple_airfoil_analysis(model)
        else:
            st.error("❌ **Model not loaded**. Please ensure 'best_airfoil_model.keras' is available.")
            st.info("💡 **What you can do:**\n- Check if the model file exists in the current directory\n- Verify the file name is exactly 'best_airfoil_model.keras'\n- Ensure the file is not corrupted")
    
    elif current_mode == 'quiz':
        # Quiz mode doesn't require the model
        run_quiz_section()
    
    # Footer with documentation and information
    st.markdown("---")
    st.markdown("## 📚 **Documentation & Information**")
    
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        with st.expander("📖 **How to Use This Application**"):
            st.markdown("""
            ### 🎯 **Single Airfoil Analysis**
            1. Choose an aircraft preset or select "Custom" for manual configuration
            2. Adjust NACA parameters using the sidebar sliders
            3. Preview the airfoil geometry
            4. Click "Predict Cp Distribution" to run the analysis
            5. Export results as CSV, text report, or PDF
            
            ### 🔄 **Multiple Airfoil Comparison**
            1. Select the number of airfoils to compare (2-4)
            2. Choose quick preset comparisons or configure manually
            3. Set up each airfoil using the configuration panels
            4. Click "Analyze All Airfoils" to run the comparison
            5. Review side-by-side visualizations and metrics
            6. Export comparison data and summary reports
            
            ### 🧠 **Knowledge Quiz**
            1. Select your difficulty level (Beginner/Intermediate/Advanced)
            2. Answer 5 multiple-choice questions
            3. Get instant feedback with detailed explanations
            4. Review your performance and learn from mistakes
            5. Progress to higher difficulty levels
            """)
    
    with col_doc2:
        with st.expander("🔬 **Technical Information**"):
            st.markdown("""
            ### 🧠 **Neural Network Model**
            - **Architecture**: Deep feedforward neural network
            - **Input**: 400 scaled airfoil coordinates (200 x,y pairs)
            - **Output**: 200 pressure coefficient values (100 upper + 100 lower surface)
            - **Training Data**: NACA 4-digit airfoil database
            - **Conditions**: Zero angle of attack, subsonic flow
            
            ### 📐 **NACA 4-Digit Code**
            - **First digit (M)**: Maximum camber as % of chord (0-7%)
            - **Second digit (P)**: Position of max camber (20-60% chord)
            - **Last two digits (XX)**: Maximum thickness as % of chord (6-30%)
            - **Example**: NACA 2412 = 2% camber at 40% chord, 12% thickness
            
            ### 📊 **Pressure Coefficient (Cp)**
            - **Definition**: (P - P∞) / (0.5 × ρ × V²)
            - **Negative values**: Suction (pressure below freestream)
            - **Positive values**: Compression (pressure above freestream)
            - **Peak suction**: Minimum Cp value (most negative)
            """)
    
    with st.expander("⚙️ **System Requirements & Setup**"):
        st.markdown("""
        ### 📋 **Required Files**
        - `best_airfoil_model.keras` - Trained neural network model
        - Python packages: streamlit, tensorflow, numpy, pandas, matplotlib, scipy
        
        ### 📦 **Optional Packages**
        - `reportlab` - For PDF report generation
        - Install with: `pip install reportlab`
        
        ### 🚀 **Running the Application**
        ```bash
        streamlit run app.py
        ```
        
        ### 🔧 **Troubleshooting**
        - **Model not found**: Ensure 'best_airfoil_model.keras' is in the same directory
        - **Import errors**: Install missing packages with pip
        - **Prediction errors**: Check that input parameters are within valid ranges
        - **PDF generation**: Install reportlab package for PDF export functionality
        """)
    
    with st.expander("📝 **About This Application**"):
        st.markdown("""
        ### 🎯 **Purpose**
        This application provides an interactive platform for analyzing NACA 4-digit airfoils using 
        machine learning predictions. It's designed for:
        - **Students** learning aerodynamics and airfoil design
        - **Engineers** performing preliminary airfoil analysis
        - **Researchers** comparing airfoil characteristics
        - **Educators** teaching aerodynamic concepts
        
        ### ✨ **Key Features**
        - **Real-time predictions** using trained neural networks
        - **Interactive visualizations** with detailed analysis
        - **Comparative analysis** of multiple airfoils
        - **Educational quiz system** with progressive difficulty
        - **Professional reporting** with export capabilities
        - **Aircraft preset configurations** for common applications
        
        ### 🔮 **Future Enhancements**
        - Support for 5-digit NACA airfoils
        - Angle of attack variations
        - Reynolds number effects
        - Compressibility corrections
        - Custom airfoil uploads
        - 3D visualization capabilities
        
        ### 📞 **Support**
        For technical support, feature requests, or educational use cases, 
        please refer to the documentation or contact the development team.
        """)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

