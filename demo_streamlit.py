""""
import streamlit as st

st.title("My Streamlit app")

name = st.text_input("Enter your name")

if st.button(" hello"):
    if name:
        st.success(f"hello {name}, welcome home ")
    else:
        st.warning("Please enter your name")
        """

import streamlit as st

# App title
st.title(" Simple Python Calculator")

# Description
st.write("This is a basic calculator built with Streamlit!")

# Take user inputs
num1 = st.number_input("Enter first number", step=1.0)
num2 = st.number_input("Enter second number", step=1.0)

# Dropdown for operation
operation = st.selectbox(
    "Choose operation",
    ("Add", "Subtract", "Multiply", "Divide")
)

# Calculate result when button clicked
if st.button("Calculate"):
    if operation == "Add":
        result = num1 + num2
        st.success(f"✅ Result: {num1} + {num2} = {result}")
    elif operation == "Subtract":
        result = num1 - num2
        st.success(f"✅ Result: {num1} - {num2} = {result}")
    elif operation == "Multiply":
        result = num1 * num2
        st.success(f"✅ Result: {num1} × {num2} = {result}")
    elif operation == "Divide":
        if num2 != 0:
            result = num1 / num2
            st.success(f"✅ Result: {num1} ÷ {num2} = {result}")
        else:
            st.error("❌ Cannot divide by zero!")


