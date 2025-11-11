import streamlit as st
import random

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Greeting Form", page_icon="👋", layout="centered")

# ---------------- CSS Styling ----------------
st.markdown(
    """
    <style>
    /* Set background gradient */
    body {
        background: linear-gradient(135deg, #74ABE2, #5563DE);
        color: #FFFFFF;
        font-family: "Segoe UI", sans-serif;
    }

    /* Title Styling */
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 700;
        color: #fff;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }

    /* Streamlit button custom */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00C6FF, #0072FF);
        color: white;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        padding: 0.6em 1.5em;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        transform: scale(1.07);
        background: linear-gradient(90deg, #0072FF, #00C6FF);
    }

    /* Custom form box styling */
    .stForm {
        background: rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }

    /* Style slider label and value */
    .stSlider label {
        color: #fff;
        font-weight: 600;
    }

    /* Streamlit slider customization */
    input[type=range]::-webkit-slider-thumb {
        background: linear-gradient(90deg, #00C6FF, #0072FF);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Title ----------------
st.markdown('<h1 class="title">👋 Greeting Form</h1>', unsafe_allow_html=True)
st.write("Enter your name and select your age to get a personalized greeting!")

# ---------------- Fun Dynamic Emoji Based on Age ----------------
def get_emoji_for_age(age):
    if age < 18:
        return random.choice(["🎈", "🧃", "🎮", "🧸"])
    elif age < 40:
        return random.choice(["🚀", "💻", "🔥", "🎉"])
    elif age < 65:
        return random.choice(["🌟", "💼", "🌈", "🌻"])
    else:
        return random.choice(["🌼", "🌙", "🌻", "🌳"])

# ---------------- Greeting Form ----------------
with st.form("greeting_form"):
    name = st.text_input("Enter your name", placeholder="Type your name here")

    # Fancy slider label with dynamic emoji
    age = st.slider("Select your age", min_value=1, max_value=120, value=25)
    emoji = get_emoji_for_age(age)
    st.markdown(f"**Age Selected:** {age} {emoji}")

    submitted = st.form_submit_button("✨ Greet Me ✨")

# ---------------- Form Submission Handling ----------------
if submitted:
    if name.strip() == "":
        st.warning("⚠️ Please enter your name before submitting.")
    else:
        greeting = f"👋 Hello, **{name.strip()}**! You are **{age} years old.** {emoji}"

        # Add a dynamic age-based message
        if age < 18:
            greeting += " Enjoy your youth and keep dreaming big! 🌈"
        elif age < 40:
            greeting += " Keep building your dreams — the world is yours! 💪"
        elif age < 65:
            greeting += " Wishing you continued success and happiness! 🌟"
        else:
            greeting += " May your wisdom continue to inspire others. 🌼"

        st.success(greeting)

        # Add confetti animation 🎉
        st.balloons()
