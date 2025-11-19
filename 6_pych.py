import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import json
import os

# Page configuration
st.set_page_config(
    page_title="Water Intake Tracker 💧",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .progress-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1976d2;
    }
    .sidebar-header {
        color: #1976d2;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .achievement-badge {
        background: linear-gradient(45deg, #4caf50, #8bc34a);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Data file path
DATA_FILE = "water_intake_data.json"

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = {}

def load_data():
    """Load data from JSON file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_data(data):
    """Save data to JSON file"""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def get_water_emoji(percentage):
    """Return appropriate emoji based on progress percentage"""
    if percentage >= 100:
        return "🏆"
    elif percentage >= 80:
        return "💪"
    elif percentage >= 60:
        return "👍"
    elif percentage >= 40:
        return "😊"
    elif percentage >= 20:
        return "🌱"
    else:
        return "💧"

def main():
    # Load existing data
    st.session_state.data = load_data()
    
    # Main header
    st.markdown('<h1 class="main-header">💧 Water Intake Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar for settings and input
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Settings & Input</div>', unsafe_allow_html=True)
        
        # Daily goal setting
        daily_goal = st.slider(
            "Daily Water Goal (Liters)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Set your daily water intake goal"
        )
        
        st.divider()
        
        # Date selection
        selected_date = st.date_input(
            "Select Date",
            value=date.today(),
            max_value=date.today(),
            help="Choose the date to log water intake"
        )
        
        # Current intake for selected date
        date_str = selected_date.strftime("%Y-%m-%d")
        current_intake = st.session_state.data.get(date_str, 0.0)
        
        st.write(f"Current intake for {selected_date}: **{current_intake:.1f}L**")
        
        # Input methods
        st.subheader("💧 Add Water Intake")
        
        # Quick add buttons - stacked vertically for better alignment
        st.write("**Quick Add:**")
        
        if st.button("💧 Glass (250ml)",  width="stretch", type="secondary"):
            st.session_state.data[date_str] = current_intake + 0.25
            save_data(st.session_state.data)
            st.rerun()
        
        if st.button("🥤 Bottle (500ml)",  width="stretch", type="secondary"):
            st.session_state.data[date_str] = current_intake + 0.5
            save_data(st.session_state.data)
            st.rerun()
        
        if st.button("🍶 Large Bottle (1L)",  width="stretch", type="secondary"):
            st.session_state.data[date_str] = current_intake + 1.0
            save_data(st.session_state.data)
            st.rerun()
        
        st.divider()
        
        # Custom amount input
        st.write("**Custom Amount:**")
        custom_amount = st.number_input(
            "Amount (Liters)",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.05,
            help="Enter custom water amount",
            label_visibility="collapsed"
        )
        
        # Action buttons in a clean layout
        if st.button("➕ Add to Total",  width="stretch", type="primary"):
            st.session_state.data[date_str] = current_intake + custom_amount
            save_data(st.session_state.data)
            st.rerun()
        
        if st.button("📝 Set as Total",  width="stretch", type="secondary"):
            st.session_state.data[date_str] = custom_amount
            save_data(st.session_state.data)
            st.rerun()
        
        st.divider()
        
        # Reset button with warning styling
        if st.button("🔄 Reset Day",  width="stretch", help="Clear all water intake for selected day"):
            if date_str in st.session_state.data:
                del st.session_state.data[date_str]
                save_data(st.session_state.data)
                st.rerun()
    
    # Main content area
    today_str = date.today().strftime("%Y-%m-%d")
    today_intake = st.session_state.data.get(today_str, 0.0)
    progress_percentage = (today_intake / daily_goal) * 100
    
    # Today's progress section
    st.subheader("📊 Today's Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💧 Water Consumed",
            value=f"{today_intake:.1f}L",
            delta=f"{today_intake - daily_goal:.1f}L from goal"
        )
    
    with col2:
        st.metric(
            label="🎯 Daily Goal",
            value=f"{daily_goal:.1f}L"
        )
    
    with col3:
        st.metric(
            label="📈 Progress",
            value=f"{progress_percentage:.1f}%"
        )
    
    with col4:
        remaining = max(0, daily_goal - today_intake)
        st.metric(
            label="⏳ Remaining",
            value=f"{remaining:.1f}L"
        )
    
    # Progress bar
    progress_bar_col1, progress_bar_col2 = st.columns([4, 1])
    
    with progress_bar_col1:
        st.progress(min(progress_percentage / 100, 1.0))
    
    with progress_bar_col2:
        emoji = get_water_emoji(progress_percentage)
        st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{emoji}</div>", unsafe_allow_html=True)
    
    # Achievement badges
    if progress_percentage >= 100:
        st.markdown('<div class="achievement-badge">🏆 Goal Achieved! Great job!</div>', unsafe_allow_html=True)
    elif progress_percentage >= 80:
        st.markdown('<div class="achievement-badge">💪 Almost there! Keep going!</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Weekly chart section
    st.subheader("📈 Weekly Hydration Chart")
    
    # Prepare weekly data
    end_date = date.today()
    start_date = end_date - timedelta(days=6)
    
    weekly_data = []
    for i in range(7):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        intake = st.session_state.data.get(date_str, 0.0)
        
        weekly_data.append({
            'Date': current_date,
            'Day': current_date.strftime("%a"),
            'Intake (L)': intake,
            'Goal': daily_goal,
            'Progress': f"{(intake/daily_goal*100):.1f}%"
        })
    
    df_weekly = pd.DataFrame(weekly_data)
    
    # Create interactive chart
    fig = go.Figure()
    
    # Add bar chart for actual intake
    fig.add_trace(go.Bar(
        x=df_weekly['Day'],
        y=df_weekly['Intake (L)'],
        name='Actual Intake',
        marker_color='lightblue',
        text=df_weekly['Intake (L)'].round(1),
        textposition='auto',
    ))
    
    # Add goal line
    fig.add_trace(go.Scatter(
        x=df_weekly['Day'],
        y=[daily_goal] * len(df_weekly),
        mode='lines',
        name='Daily Goal',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="Weekly Water Intake Progress",
        xaxis_title="Day of Week",
        yaxis_title="Water Intake (Liters)",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Weekly statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_intake = df_weekly['Intake (L)'].mean()
        st.metric("📊 Weekly Average", f"{avg_intake:.1f}L")
    
    with col2:
        days_achieved = sum(1 for intake in df_weekly['Intake (L)'] if intake >= daily_goal)
        st.metric("🎯 Goals Achieved", f"{days_achieved}/7 days")
    
    with col3:
        total_intake = df_weekly['Intake (L)'].sum()
        st.metric("💧 Total Weekly Intake", f"{total_intake:.1f}L")
    
    # Data table
    with st.expander("📋 View Weekly Data"):
        st.dataframe(
            df_weekly[['Day', 'Date', 'Intake (L)', 'Progress']],
            width="stretch",
            hide_index=True
        )
    
    # Tips section
    with st.expander("💡 Hydration Tips"):
        tips = [
            "🌅 Start your day with a glass of water",
            "⏰ Set reminders throughout the day",
            "🍋 Add lemon or cucumber for flavor",
            "🚰 Keep a water bottle nearby",
            "🥗 Eat water-rich foods like fruits and vegetables",
            "☕ Limit caffeine and alcohol intake",
            "🏃‍♂️ Increase intake during exercise",
            "🌡️ Drink more in hot weather"
        ]
        
        for tip in tips:
            st.write(tip)

if __name__ == "__main__":
    main()