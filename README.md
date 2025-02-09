# Enhanced Black-Scholes Option Pricing Model

## 📌 Overview
This interactive application calculates European call and put option prices using the **Black-Scholes model** and computes the option Greeks. It also provides:
- An **interactive plot** of option price vs. volatility.
- A **Monte Carlo simulation** to estimate the option price and visualize the distribution of payoffs.

The app is built with **Streamlit** and uses **NumPy, SciPy, and Plotly** for calculations and visualization.

---

## 🚀 Features
- **Black-Scholes Pricing Model**: Computes call and put option prices.
- **Option Greeks Calculation**: Provides Delta, Gamma, Vega, Theta, and Rho.
- **Volatility Sensitivity Analysis**: Interactive plot of option price vs. volatility.
- **Monte Carlo Simulation**: Estimates the option price using simulated price paths.
- **Interactive UI**: Adjustable inputs for underlying price, strike price, risk-free rate, volatility, and maturity.
- **Reproducibility**: Allows setting a random seed for consistent Monte Carlo results.

---

## 🛠️ Installation
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/black-scholes-app.git
cd black-scholes-app
```

### 2️⃣ **Install Dependencies**
Ensure you have Python installed (preferably 3.8+). Then, install the required packages:
```bash
pip install -r requirements.txt
```

Alternatively, you can install dependencies manually:
```bash
pip install streamlit numpy scipy plotly
```

---

## ▶️ Running the App
Once installed, run the Streamlit app with:
```bash
streamlit run app.py
```

This will launch the app in your browser.

---

## 📊 How It Works
### **1️⃣ Option Pricing (Black-Scholes Model)**
- Uses the Black-Scholes formula to compute **call and put option prices**.
- Calculates **option Greeks**: Delta, Gamma, Vega, Theta, and Rho.

### **2️⃣ Price vs. Volatility Plot**
- Generates an interactive **Plotly chart** showing how option prices change with volatility.

### **3️⃣ Monte Carlo Simulation**
- Simulates **thousands of possible price paths** to estimate the option price.
- Plots a **histogram of payoffs at maturity**.

---

## 📜 Black-Scholes Formula
The Black-Scholes equation for a **European Call Option**:
\[ C = S N(d_1) - K e^{-rT} N(d_2) \]

For a **Put Option**:
\[ P = K e^{-rT} N(-d_2) - S N(-d_1) \]

where:
\[ d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}} \]
\[ d_2 = d_1 - \sigma \sqrt{T} \]

---

## ⚡ Technologies Used
- **Streamlit** – Interactive UI framework
- **NumPy** – Numerical calculations
- **SciPy** – Probability and statistical functions
- **Plotly** – Interactive graphs

---

## 📌 Contributing
Contributions are welcome! To contribute:
1. **Fork** the repository.
2. **Create** a feature branch (`git checkout -b feature-name`).
3. **Commit** your changes (`git commit -m "Added new feature"`).
4. **Push** to your branch (`git push origin feature-name`).
5. Open a **Pull Request** on GitHub.

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 📧 Contact
For questions or feedback, reach out via **GitHub Issues** or email at [your.email@example.com](mailto:your.email@example.com).

