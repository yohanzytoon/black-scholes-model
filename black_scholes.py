import streamlit as st
import numpy as np
import math
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# Page Configuration & Custom CSS
# ---------------------------
st.set_page_config(page_title="Enhanced Black-Scholes Option Pricing", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #F5F5F5; }
    .stButton>button {background-color: #4CAF50; color: white; font-size: 16px;}
    </style>
    """, unsafe_allow_html=True)

st.title("Enhanced Black-Scholes Option Pricing Model")

# ---------------------------
# Sidebar: Input Parameters
# ---------------------------
st.sidebar.header("Input Parameters")

S = st.sidebar.number_input("Underlying Price (S)", value=100.0, min_value=0.01, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (T, in years)", value=1.0, min_value=0.01, format="%.2f")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, min_value=0.001, format="%.4f")
option_type = st.sidebar.radio("Option Type", ("Call", "Put"))

st.sidebar.markdown("---")
st.sidebar.header("Plot Settings")
vol_min = st.sidebar.number_input("Min Volatility", value=0.05, min_value=0.001, format="%.4f")
vol_max = st.sidebar.number_input("Max Volatility", value=1.0, min_value=0.001, format="%.4f")
num_points = st.sidebar.slider("Number of Points", min_value=50, max_value=500, value=100)

if vol_min >= vol_max:
    st.sidebar.error("Min Volatility must be less than Max Volatility.")

st.sidebar.markdown("---")
st.sidebar.header("Monte Carlo Settings")
num_simulations = st.sidebar.number_input("Number of Simulations", value=10000, min_value=1000, step=1000)

# Option to fix the random seed for reproducibility
fix_seed = st.sidebar.checkbox("Fix random seed for reproducibility", value=True)
seed_value = st.sidebar.number_input("Random Seed (if fixed)", value=42, step=1) if fix_seed else None

# ---------------------------
# Black-Scholes Pricing & Greeks Functions (with caching)
# ---------------------------
@st.cache_data
def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
    return call

@st.cache_data
def black_scholes_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put = K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put

@st.cache_data
def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = stats.norm.pdf(d1)
    if option_type == "Call":
        delta = stats.norm.cdf(d1)
    else:
        delta = stats.norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * pdf_d1
    if option_type == "Call":
        theta = (-S * sigma * pdf_d1 / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * stats.norm.cdf(d2))
        rho = K * T * math.exp(-r * T) * stats.norm.cdf(d2)
    else:
        theta = (-S * sigma * pdf_d1 / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * stats.norm.cdf(-d2))
        rho = -K * T * math.exp(-r * T) * stats.norm.cdf(-d2)
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

# ---------------------------
# Monte Carlo Simulation Function
# ---------------------------
def monte_carlo_option_price(S, K, T, r, sigma, option_type, n_simulations, seed=None):
    # Set seed if provided for reproducibility
    if seed is not None:
        np.random.seed(int(seed))
    else:
        np.random.seed(None)
    Z = np.random.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    if option_type == "Call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    price_estimate = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    return price_estimate, payoffs, std_error

# ---------------------------
# Main Tabs Layout
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Option Calculator", "Price vs. Volatility", "Monte Carlo Simulation"])

# --- Tab 1: Option Calculator ---
with tab1:
    st.header("Option Calculator")
    # Calculate analytical option price using Black–Scholes formula
    if option_type == "Call":
        price = black_scholes_call(S, K, T, r, sigma)
    else:
        price = black_scholes_put(S, K, T, r, sigma)
    
    st.subheader("Option Price")
    st.markdown(f"**{option_type} Option Price:** {price:.4f}")
    
    st.subheader("Option Greeks")
    greeks = calculate_greeks(S, K, T, r, sigma, option_type)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Delta:** {greeks['Delta']:.4f}")
        st.markdown(f"**Gamma:** {greeks['Gamma']:.4f}")
        st.markdown(f"**Vega:**  {greeks['Vega']:.4f}")
    with col2:
        st.markdown(f"**Theta:** {greeks['Theta']:.4f}")
        st.markdown(f"**Rho:**   {greeks['Rho']:.4f}")

# --- Tab 2: Price vs. Volatility Plot ---
with tab2:
    st.header("Option Price vs. Volatility")
    if vol_min < vol_max:
        sigma_range = np.linspace(vol_min, vol_max, num_points)
        prices = []
        for s_val in sigma_range:
            if option_type == "Call":
                prices.append(black_scholes_call(S, K, T, r, s_val))
            else:
                prices.append(black_scholes_put(S, K, T, r, s_val))
        
        # Create an interactive Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sigma_range, y=prices,
                                 mode='lines',
                                 name=f"{option_type} Price",
                                 line=dict(color="blue", width=2)))
        fig.update_layout(title="Option Price vs. Volatility",
                          xaxis_title="Volatility (σ)",
                          yaxis_title="Option Price",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please ensure that Min Volatility is less than Max Volatility.")

# --- Tab 3: Monte Carlo Simulation ---
with tab3:
    st.header("Monte Carlo Simulation")
    st.markdown("Simulate the option price using Monte Carlo simulation and compare it with the analytical Black–Scholes price.")
    
    if st.button("Run Monte Carlo Simulation"):
        mc_price, payoffs, std_error = monte_carlo_option_price(
            S, K, T, r, sigma, option_type, int(num_simulations), seed_value
        )
        st.markdown(f"**Monte Carlo Estimated {option_type} Option Price:** {mc_price:.4f}")
        st.markdown(f"**Standard Error:** {std_error:.4f}")
        st.markdown(f"**Black-Scholes {option_type} Option Price:** {price:.4f}")
        
        # Plot a histogram of the simulated payoffs
        fig2 = px.histogram(payoffs, nbins=50,
                            title="Histogram of Option Payoffs at Maturity",
                            labels={'value': 'Payoff'},
                            template="plotly_white")
        fig2.update_layout(xaxis_title="Payoff", yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Click the button above to run the Monte Carlo simulation.")

# ---------------------------
# About Section (Expandable)
# ---------------------------
with st.expander("About This App"):
    st.markdown("""
    **Black-Scholes Option Pricing Model**  
   
    This interactive application calculates European call and put option prices using the Black–Scholes formula and computes the option Greeks (Delta, Gamma, Vega, Theta, and Rho). It also provides:
    
    - An interactive plot of the option price versus volatility (with adjustable settings).
    - A Monte Carlo simulation to estimate the option price and visualize the distribution of payoffs.
    
    **Features:**  
    - Adjustable inputs for the underlying asset, strike price, time to maturity, risk-free rate, and volatility.
    - An option to fix the random seed for reproducibility in Monte Carlo simulations.
    - Interactive plots powered by Plotly.
    
    **Technologies used:**  
    - [Streamlit](https://streamlit.io/)  
    - [NumPy](https://numpy.org/)  
    - [SciPy](https://www.scipy.org/)  
    - [Plotly](https://plotly.com/)
    """)
