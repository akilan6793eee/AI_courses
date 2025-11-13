import streamlit as st
import math

def main():
    st.title("🧮 Calculator App")
    st.write("Perform basic arithmetic operations")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            num1 = st.number_input("Enter first number:", value=0.0, format="%.2f")
        except Exception as e:
            st.error(f"Invalid input for first number: {e}")
            num1 = 0.0
    
    with col2:
        try:
            num2 = st.number_input("Enter second number:", value=0.0, format="%.2f")
        except Exception as e:
            st.error(f"Invalid input for second number: {e}")
            num2 = 0.0
    
    # Update session state with current values
    st.session_state.num1 = num1
    st.session_state.num2 = num2
    
    # Operation selection
    operation = st.selectbox(
        "Select operation:",
        ["Addition (+)", "Subtraction (-)", "Multiplication (×)", "Division (÷)"]
    )
    
    # Calculate button
    if st.button("Calculate", type="primary"):
        try:
            result = perform_calculation(num1, num2, operation)
            if result is not None:
                st.success(f"**Result: {result}**")
                
                # Display the calculation
                op_symbol = get_operation_symbol(operation)
                st.info(f"Calculation: {num1} {op_symbol} {num2} = {result}")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    
    # Clear button
    if st.button("Clear"):
        st.rerun()
    
    # Add some helpful information
    with st.expander("ℹ️ Calculator Features"):
        st.write("""
        - **Addition**: Add two numbers together
        - **Subtraction**: Subtract the second number from the first
        - **Multiplication**: Multiply two numbers
        - **Division**: Divide the first number by the second (with zero-division protection)
        - **Error Handling**: Handles invalid inputs and division by zero
        - **Precision**: Supports decimal numbers up to 6 decimal places
        """)

def perform_calculation(num1, num2, operation):
    """
    Perform the selected calculation with error handling
    
    Args:
        num1 (float): First number
        num2 (float): Second number
        operation (str): Selected operation
    
    Returns:
        float: Result of the calculation, or None if error
    """
    try:
        if operation == "Addition (+)":
            return num1 + num2
            
        elif operation == "Subtraction (-)":
            return num1 - num2
            
        elif operation == "Multiplication (×)":
            return num1 * num2
            
        elif operation == "Division (÷)":
            if num2 == 0:
                st.error("❌ Error: Division by zero is not allowed!")
                return None
            return num1 / num2
            
        else:
            st.error("❌ Invalid operation selected!")
            return None
            
    except OverflowError:
        st.error("❌ Error: Result is too large to calculate!")
        return None
    except ValueError as e:
        st.error(f"❌ Error: Invalid number format - {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error during calculation: {e}")
        return None

def get_operation_symbol(operation):
    """Get the mathematical symbol for display"""
    symbols = {
        "Addition (+)": "+",
        "Subtraction (-)": "-",
        "Multiplication (×)": "×",
        "Division (÷)": "÷"
    }
    return symbols.get(operation, "?")

if __name__ == "__main__":
    main()