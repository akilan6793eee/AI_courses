import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="BMI Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .bmi-value {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .category-text {
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚖️ Interactive BMI Calculator</h1>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.markdown("## 📏 Your Measurements")

# Unit selection
col1, col2 = st.sidebar.columns(2)
with col1:
    height_unit = st.selectbox("Height Unit", ["cm", "ft/in"], key="height_unit")
with col2:
    weight_unit = st.selectbox("Weight Unit", ["kg", "lbs"], key="weight_unit")

st.sidebar.markdown("---")

# Height input based on unit
if height_unit == "cm":
    height_cm = st.sidebar.slider(
        "Height (cm)", 
        min_value=100, 
        max_value=250, 
        value=170, 
        step=1,
        help="Adjust your height in centimeters"
    )
    height_in_cm = height_cm
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        feet = st.slider("Feet", min_value=3, max_value=8, value=5, step=1)
    with col2:
        inches = st.slider("Inches", min_value=0, max_value=11, value=7, step=1)
    height_in_cm = (feet * 12 + inches) * 2.54

# Weight input based on unit
if weight_unit == "kg":
    weight_kg = st.sidebar.slider(
        "Weight (kg)", 
        min_value=30, 
        max_value=200, 
        value=70, 
        step=1,
        help="Adjust your weight in kilograms"
    )
    weight_in_kg = weight_kg
else:
    weight_lbs = st.sidebar.slider(
        "Weight (lbs)", 
        min_value=66, 
        max_value=440, 
        value=154, 
        step=1,
        help="Adjust your weight in pounds"
    )
    weight_in_kg = weight_lbs * 0.453592

# Calculate BMI
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#74C0FC", "#1864AB"
    elif 18.5 <= bmi < 25:
        return "Normal Weight", "#69DB7C", "#2B8A3E"
    elif 25 <= bmi < 30:
        return "Overweight", "#FFD43B", "#E8590C"
    else:
        return "Obese", "#FF8787", "#C92A2A"

# Calculate current BMI
current_bmi = calculate_bmi(weight_in_kg, height_in_cm)
category, bg_color, text_color = get_bmi_category(current_bmi)

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # BMI Display
    st.markdown(
        f'<div class="bmi-value" style="background-color: {bg_color}; color: {text_color};">'
        f'{current_bmi}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="category-text" style="color: {text_color};">{category}</div>',
        unsafe_allow_html=True
    )

# Create BMI gauge chart
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = current_bmi,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "BMI Gauge"},
    delta = {'reference': 22.5, 'position': "top"},
    gauge = {
        'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': text_color},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 18.5], 'color': '#E3F2FD'},
            {'range': [18.5, 25], 'color': '#E8F5E8'},
            {'range': [25, 30], 'color': '#FFF3E0'},
            {'range': [30, 40], 'color': '#FFEBEE'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': current_bmi
        }
    }
))

fig_gauge.update_layout(
    height=300,
    font={'color': "darkblue", 'family': "Arial"}
)

# Display sections
col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Display current measurements
    st.markdown("### 📊 Your Current Stats")
    if height_unit == "cm":
        height_display = f"{height_in_cm:.0f} cm"
    else:
        height_display = f"{feet}' {inches}\""
    
    if weight_unit == "kg":
        weight_display = f"{weight_in_kg:.1f} kg"
    else:
        weight_display = f"{weight_lbs} lbs"
    
    st.markdown(f"""
    <div class="metric-card">
        <strong>Height:</strong> {height_display}<br>
        <strong>Weight:</strong> {weight_display}<br>
        <strong>BMI:</strong> {current_bmi}
    </div>
    """, unsafe_allow_html=True)

with col2:
    # BMI Categories Chart
    categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
    ranges = ['< 18.5', '18.5 - 24.9', '25.0 - 29.9', '≥ 30.0']
    colors = ['#74C0FC', '#69DB7C', '#FFD43B', '#FF8787']
    
    # Highlight current category
    highlight_colors = []
    for i, cat in enumerate(categories):
        if cat.replace(' Weight', '').replace(' ', ' Weight') == category:
            highlight_colors.append(colors[i])
        else:
            highlight_colors.append('#E0E0E0')
    
    fig_bar = px.bar(
        x=categories,
        y=[18.5, 6.4, 4.9, 10],  # Range sizes for visualization
        color=categories,
        color_discrete_sequence=highlight_colors,
        title="BMI Categories"
    )
    
    fig_bar.update_layout(
        showlegend=False,
        height=300,
        yaxis_title="BMI Range Size",
        xaxis_title="Categories"
    )
    
    # Add range annotations
    for i, (cat, range_text) in enumerate(zip(categories, ranges)):
        fig_bar.add_annotation(
            x=i,
            y=[18.5, 6.4, 4.9, 10][i]/2,
            text=range_text,
            showarrow=False,
            font=dict(color="white", size=12, family="Arial Black")
        )
    
    st.plotly_chart(fig_bar, use_container_width=True)

# Health recommendations based on BMI category
st.markdown("---")
st.markdown("### 💡 Health Insights & Recommendations")

recommendations = {
    "Underweight": {
        "icon": "🍎",
        "text": "Consider consulting a healthcare provider about healthy weight gain strategies. Focus on nutrient-rich foods and strength training.",
        "tips": ["Eat frequent, balanced meals", "Include healthy fats", "Consider strength training", "Consult a nutritionist"]
    },
    "Normal Weight": {
        "icon": "✅",
        "text": "Great job! You're in the healthy weight range. Maintain your current lifestyle with regular exercise and balanced nutrition.",
        "tips": ["Maintain regular exercise", "Keep eating balanced meals", "Stay hydrated", "Get adequate sleep"]
    },
    "Overweight": {
        "icon": "⚠️",
        "text": "Consider adopting healthier eating habits and increasing physical activity. Small changes can make a big difference!",
        "tips": ["Increase daily activity", "Focus on portion control", "Choose whole foods", "Track your progress"]
    },
    "Obese": {
        "icon": "🏥",
        "text": "It's recommended to consult with a healthcare provider for a personalized weight management plan and health assessment.",
        "tips": ["Consult healthcare provider", "Start with small changes", "Focus on sustainable habits", "Consider professional support"]
    }
}

rec = recommendations[category]
st.markdown(f"""
<div class="info-box">
    <h4>{rec['icon']} {category} - Health Guidance</h4>
    <p>{rec['text']}</p>
</div>
""", unsafe_allow_html=True)

# Tips in columns
st.markdown("#### 🎯 Quick Tips:")
tip_cols = st.columns(2)
for i, tip in enumerate(rec['tips']):
    with tip_cols[i % 2]:
        st.markdown(f"• {tip}")

# BMI Calculator Formula Information
with st.expander("ℹ️ About BMI Calculation"):
    st.markdown("""
    **BMI Formula:** BMI = Weight (kg) / Height (m)²
    
    **BMI Categories (Adult):**
    - **Underweight:** Below 18.5
    - **Normal weight:** 18.5–24.9  
    - **Overweight:** 25–29.9
    - **Obese:** 30 or greater
    
    **Note:** BMI is a screening tool and doesn't diagnose body fatness or health. 
    It doesn't account for muscle mass, bone density, overall body composition, and racial/ethnic differences.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>⚠️ This BMI calculator is for educational purposes only. Always consult with healthcare professionals for personalized health advice.</small>
</div>
""", unsafe_allow_html=True)