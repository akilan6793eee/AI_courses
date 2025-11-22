# enhanced_currency_converter_fixed.py
# Full fixed Streamlit app with dropdown visibility fixes and CSS f-string issue resolved.
# Uses the uploaded file path as page_icon (developer instruction).
# Run: streamlit run enhanced_currency_converter_fixed.py

import streamlit as st
import requests
import json
from datetime import datetime
import time

# -------------------------
# Page configuration
# -------------------------
# Use the uploaded file path as the page icon (per your instruction)
SAMPLE_FILE_PATH = "/mnt/data/WhatsApp Video 2025-02-14 at 7.19.12 PM.mp4"

st.set_page_config(
    page_title="Currency Converter Pro",
    page_icon=SAMPLE_FILE_PATH,  # local path used as icon per developer instruction
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    ...
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.2), rgba(255, 107, 149, 0.2));
        backdrop-filter: blur(5px);
        border-left: 4px solid #dc3545;
        border-radius: 10px;
    }

    /* Make ALL Streamlit buttons white */
    .stButton > button {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 2px solid #e2e8f0 !important;
        padding: 10px 20px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #f1f5f9 !important;
        color: #020617 !important;
        border-color: #cbd5e1 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.18) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) scale(0.98);
        box-shadow: 0 2px 8px rgba(0,0,0,0.10) !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Currency data with flags and names
# -------------------------
CURRENCIES = {
    'USD': {'name': 'US Dollar', 'flag': '🇺🇸'},
    'EUR': {'name': 'Euro', 'flag': '🇪🇺'},
    'GBP': {'name': 'British Pound', 'flag': '🇬🇧'},
    'JPY': {'name': 'Japanese Yen', 'flag': '🇯🇵'},
    'AUD': {'name': 'Australian Dollar', 'flag': '🇦🇺'},
    'CAD': {'name': 'Canadian Dollar', 'flag': '🇨🇦'},
    'CHF': {'name': 'Swiss Franc', 'flag': '🇨🇭'},
    'CNY': {'name': 'Chinese Yuan', 'flag': '🇨🇳'},
    'SEK': {'name': 'Swedish Krona', 'flag': '🇸🇪'},
    'NZD': {'name': 'New Zealand Dollar', 'flag': '🇳🇿'},
    'MXN': {'name': 'Mexican Peso', 'flag': '🇲🇽'},
    'SGD': {'name': 'Singapore Dollar', 'flag': '🇸🇬'},
    'HKD': {'name': 'Hong Kong Dollar', 'flag': '🇭🇰'},
    'NOK': {'name': 'Norwegian Krone', 'flag': '🇳🇴'},
    'KRW': {'name': 'South Korean Won', 'flag': '🇰🇷'},
    'TRY': {'name': 'Turkish Lira', 'flag': '🇹🇷'},
    'RUB': {'name': 'Russian Ruble', 'flag': '🇷🇺'},
    'INR': {'name': 'Indian Rupee', 'flag': '🇮🇳'},
    'BRL': {'name': 'Brazilian Real', 'flag': '🇧🇷'},
    'ZAR': {'name': 'South African Rand', 'flag': '🇿🇦'}
}

# -------------------------
# API helper
# -------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_exchange_rate(from_currency, to_currency):
    """
    Fetch exchange rate from a free API service
    """
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if to_currency in data.get('rates', {}):
            return data['rates'][to_currency], data.get('date', '')
        else:
            return None, None
    except Exception as e:
        st.error(f"Error fetching exchange rate: {str(e)}")
        return None, None

def format_currency_display(code):
    """Format currency for display in dropdown"""
    meta = CURRENCIES.get(code, {"name": code, "flag": ""})
    return f"{meta['flag']} {code} - {meta['name']}"

# -------------------------
# App layout + logic
# -------------------------
def main():
    # header with left logo badge + text
    st.markdown(
        """
        <div class="header-wrapper">
            <div class="logo-badge"><span class="icon">💱</span></div>
            <div>
                <h1 class="main-header">Currency Converter Pro</h1>
                <p class="header-sub">Convert currencies with real-time exchange rates — improved UI</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Create columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="converter-box">', unsafe_allow_html=True)
        
        # Currency selection
        col_from, col_to = st.columns(2)
        
        with col_from:
            st.markdown('<p style="color: #dbeafe; font-weight: 700; margin-bottom:6px;">From Currency:</p>', unsafe_allow_html=True)
            from_currency = st.selectbox(
                "Select source currency",
                options=list(CURRENCIES.keys()),
                format_func=format_currency_display,
                index=0,
                label_visibility="collapsed"
            )
        
        with col_to:
            st.markdown('<p style="color: #dbeafe; font-weight: 700; margin-bottom:6px;">To Currency:</p>', unsafe_allow_html=True)
            to_currency = st.selectbox(
                "Select target currency",
                options=list(CURRENCIES.keys()),
                format_func=format_currency_display,
                index=1,
                label_visibility="collapsed"
            )
        
        # Amount input
        st.markdown('<p style="color: #dbeafe; font-weight: 700; margin-bottom:6px;">Amount:</p>', unsafe_allow_html=True)
        amount = st.number_input(
            "Enter amount to convert",
            min_value=0.01,
            value=100.0,
            step=0.01,
            format="%.2f",
            label_visibility="collapsed"
        )
        
        # Swap button
        col_swap1, col_swap2, col_swap3 = st.columns([1, 1, 1])
        with col_swap2:
            swap_clicked = st.button("🔄 Swap Currencies")
            if swap_clicked:
                st.session_state.temp_from = to_currency
                st.session_state.temp_to = from_currency
                # modern Streamlit rerun
                st.rerun()
        
        # Handle currency swapping
        if hasattr(st.session_state, 'temp_from'):
            from_currency = st.session_state.temp_from
            to_currency = st.session_state.temp_to
            del st.session_state.temp_from
            del st.session_state.temp_to
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Convert button and results
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        convert_clicked = st.button("💰 Convert Currency")
        if convert_clicked:
            if from_currency == to_currency:
                st.warning("⚠️ Please select different currencies for conversion.")
            else:
                with st.spinner("Fetching latest exchange rates..."):
                    rate, date = get_exchange_rate(from_currency, to_currency)
                    
                    if rate:
                        converted_amount = amount * rate
                        
                        # Display result (refreshed design)
                        st.markdown(f'''
                        <div class="result-box">
                            <div style="font-size:1.1rem;opacity:0.9">{CURRENCIES[from_currency]['flag']} {amount:.2f} {from_currency} → {CURRENCIES[to_currency]['flag']} {to_currency}</div>
                            <div style="font-size:1.6rem;margin-top:6px">{converted_amount:.2f} <span style="opacity:0.8;font-size:0.8rem">{to_currency}</span></div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Display rate information
                        st.markdown(f'''
                        <div class="rate-info">
                            <strong>📊 Exchange Rate Information:</strong><br>
                            1 {from_currency} = {rate:.6f} {to_currency}<br>
                            1 {to_currency} = {1/rate:.6f} {from_currency}<br>
                            <small>📅 Last updated: {date}</small>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Additional conversion amounts
                        st.markdown('<p style="color: #dbeafe; font-weight: 700; margin-top:8px; margin-bottom:6px;">💡 Quick Reference:</p>', unsafe_allow_html=True)
                        reference_amounts = [1, 10, 100, 1000]
                        cols = st.columns(len(reference_amounts))
                        
                        for i, ref_amount in enumerate(reference_amounts):
                            with cols[i]:
                                ref_converted = ref_amount * rate
                                st.metric(
                                    label=f"{ref_amount} {from_currency}",
                                    value=f"{ref_converted:.2f} {to_currency}"
                                )
                    else:
                        st.error("❌ Unable to fetch exchange rate. Please try again later.")
    
    # Historical rate chart placeholder
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='background-color:rgba(255,255,255,0.02);padding:12px;border-radius:8px;color:#cfe9ff;'>"
            "📈 <b>Pro Tip:</b> Exchange rates fluctuate throughout the day. For large transactions, consider the timing of your conversion!"
            "</div>",
            unsafe_allow_html=True
        )
    
    # Footer
    st.markdown('''
    <div class="footer">
        <p>💡 Powered by real-time exchange rate APIs | Built with Streamlit</p>
        <p>⚠️ Disclaimer: Rates are for informational purposes only. Please verify with your bank for actual transaction rates.</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
