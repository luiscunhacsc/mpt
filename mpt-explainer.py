import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import datetime

#######################################
# 1) Define callback functions:
#    - One to reset defaults
#    - One to set Lab parameters
#######################################
def reset_parameters():
    st.session_state["stock1_return_slider"] = 5.0
    st.session_state["stock1_risk_slider"] = 15.0
    st.session_state["stock2_return_slider"] = 10.0
    st.session_state["stock2_risk_slider"] = 25.0
    st.session_state["correlation_slider"] = 0.3
    st.session_state["risk_free_rate_slider"] = 3.0
    st.session_state["num_portfolios_slider"] = 1000

def set_low_correlation_parameters():
    st.session_state["stock1_return_slider"] = 5.0
    st.session_state["stock1_risk_slider"] = 15.0
    st.session_state["stock2_return_slider"] = 10.0
    st.session_state["stock2_risk_slider"] = 25.0
    st.session_state["correlation_slider"] = -0.2
    st.session_state["risk_free_rate_slider"] = 3.0
    st.session_state["num_portfolios_slider"] = 1000

def set_high_correlation_parameters():
    st.session_state["stock1_return_slider"] = 5.0
    st.session_state["stock1_risk_slider"] = 15.0
    st.session_state["stock2_return_slider"] = 10.0
    st.session_state["stock2_risk_slider"] = 25.0
    st.session_state["correlation_slider"] = 0.8
    st.session_state["risk_free_rate_slider"] = 3.0
    st.session_state["num_portfolios_slider"] = 1000

def set_negative_correlation_parameters():
    st.session_state["stock1_return_slider"] = 5.0
    st.session_state["stock1_risk_slider"] = 15.0
    st.session_state["stock2_return_slider"] = 10.0
    st.session_state["stock2_risk_slider"] = 25.0
    st.session_state["correlation_slider"] = -0.7
    st.session_state["risk_free_rate_slider"] = 3.0
    st.session_state["num_portfolios_slider"] = 1000

def set_multiple_assets_parameters():
    st.session_state["stock1_return_slider"] = 5.0
    st.session_state["stock1_risk_slider"] = 15.0
    st.session_state["stock2_return_slider"] = 10.0
    st.session_state["stock2_risk_slider"] = 25.0
    st.session_state["correlation_slider"] = 0.3
    st.session_state["risk_free_rate_slider"] = 3.0
    st.session_state["num_portfolios_slider"] = 1000
    # In multi-asset mode, we'll use preset parameters for other assets
#######################################

# MPT calculation functions
def calculate_portfolio_performance(weights, returns, cov_matrix):
    """Calculate expected portfolio return and risk"""
    portfolio_return = np.sum(returns * weights)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_stddev

def generate_random_portfolios(num_portfolios, returns, cov_matrix, num_assets):
    """Generate random portfolio weights and calculate performance"""
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_stddev = calculate_portfolio_performance(weights, returns, cov_matrix)
        
        results[0,i] = portfolio_stddev
        results[1,i] = portfolio_return
        # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        results[2,i] = (portfolio_return - st.session_state.risk_free_rate_slider/100) / portfolio_stddev
    
    return results, weights_record

def find_optimal_portfolios(returns, cov_matrix, num_assets):
    """Find minimum variance and tangency (maximum Sharpe ratio) portfolios"""
    from scipy.optimize import minimize
    
    # For minimum variance portfolio
    def min_variance_objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # For maximum Sharpe ratio portfolio
    def neg_sharpe_ratio(weights):
        portfolio_return, portfolio_stddev = calculate_portfolio_performance(weights, returns, cov_matrix)
        sharpe_ratio = (portfolio_return - st.session_state.risk_free_rate_slider/100) / portfolio_stddev
        return -sharpe_ratio
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess (equal weights)
    initial_guess = np.array([1/num_assets] * num_assets)
    
    # Optimize for minimum variance
    min_var_result = minimize(min_variance_objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    min_var_weights = min_var_result['x']
    min_var_return, min_var_stddev = calculate_portfolio_performance(min_var_weights, returns, cov_matrix)
    
    # Optimize for maximum Sharpe ratio
    max_sharpe_result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    max_sharpe_weights = max_sharpe_result['x']
    max_sharpe_return, max_sharpe_stddev = calculate_portfolio_performance(max_sharpe_weights, returns, cov_matrix)
    
    return {
        'min_var': {
            'weights': min_var_weights,
            'return': min_var_return,
            'risk': min_var_stddev
        },
        'max_sharpe': {
            'weights': max_sharpe_weights,
            'return': max_sharpe_return,
            'risk': max_sharpe_stddev
        }
    }

def calculate_efficient_frontier(returns, cov_matrix, num_assets, points=100):
    """Calculate the efficient frontier by targeting returns"""
    from scipy.optimize import minimize
    
    # Get minimum variance and maximum return portfolios
    min_var_port = find_optimal_portfolios(returns, cov_matrix, num_assets)['min_var']
    
    # Find the return of the highest returning asset
    max_return = max(returns)
    
    # Generate range of target returns
    target_returns = np.linspace(min_var_port['return'], max_return, points)
    efficient_risk = []
    efficient_return = []
    
    # For each target return, find the minimum variance portfolio
    for target in target_returns:
        # Objective function (minimize variance for a given target return)
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Target return constraint
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * returns) - target}
        )
        
        bounds = tuple((0, 1) for asset in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result['success']:
            efficient_risk.append(result['fun'])
            efficient_return.append(target)
    
    return np.array(efficient_risk), np.array(efficient_return)

# Configure the Streamlit app
st.set_page_config(layout="wide", page_title="Modern Portfolio Theory Explainer")
st.title("📊 Understanding Modern Portfolio Theory")
st.markdown("Explore the foundational concepts of MPT, risk-return relationships, diversification benefits, and portfolio optimization.")

# Sidebar for input parameters
with st.sidebar:
    st.header("⚙️ Parameters")
    
    st.button("↺ Reset Parameters", on_click=reset_parameters)

    stock1_return = st.slider("Stock 1 Expected Return (%)", 0.0, 20.0, 5.0, key='stock1_return_slider')
    stock1_risk = st.slider("Stock 1 Risk (Volatility %)", 5.0, 50.0, 15.0, key='stock1_risk_slider')
    
    stock2_return = st.slider("Stock 2 Expected Return (%)", 0.0, 20.0, 10.0, key='stock2_return_slider')
    stock2_risk = st.slider("Stock 2 Risk (Volatility %)", 5.0, 50.0, 25.0, key='stock2_risk_slider')
    
    correlation = st.slider("Correlation between Stocks", -1.0, 1.0, 0.3, 0.1, key='correlation_slider')
    
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 3.0, 0.1, key='risk_free_rate_slider')
    
    num_portfolios = st.slider("Number of Random Portfolios", 100, 5000, 1000, 100, key='num_portfolios_slider')
    
    # Disclaimer and license
    st.markdown("---")
    st.markdown(
    """
    **⚠️ Disclaimer**  
    *Educational purposes only. No accuracy guarantees. Do not use as investment advice.*  
    
    <small>
    The author does not provide investment advice and does not endorse any particular investment strategy. 
    All information provided is for educational purposes only and should not be construed as financial or 
    investment advice. Investing involves significant risks and may not be suitable for all investors. 
    Always consult a qualified financial professional before making any investment decisions.
    </small>
    """,
    unsafe_allow_html=True
    )

    
    st.markdown("""
    <div style="margin-top: 20px;">
        <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en" target="_blank">
            <img src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" alt="CC BY-NC 4.0">
        </a>
        <br>
        <span style="font-size: 0.8em;">By Luís Simões da Cunha, 2025</span>
    </div>
    """, unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Interactive Portfolio Tool", 
    "📚 MPT Theory", 
    "📖 Comprehensive Tutorial", 
    "🛠️ Practical Labs",
    "🧮 Playground"
])

with tab1:
    # Calculate and display portfolio optimization based on inputs
    # Set up the assets' expected returns and covariance matrix
    num_assets = 2  # For simplicity, we start with 2 assets
    
    returns = np.array([stock1_return/100, stock2_return/100])
    
    # Build a covariance matrix
    vol = np.array([stock1_risk/100, stock2_risk/100])
    corr_matrix = np.array([[1, correlation], [correlation, 1]])
    cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)
    
    # Generate random portfolios
    results, weights_record = generate_random_portfolios(num_portfolios, returns, cov_matrix, num_assets)
    
    # Find optimal portfolios
    optimal_portfolios = find_optimal_portfolios(returns, cov_matrix, num_assets)
    min_var_portfolio = optimal_portfolios['min_var']
    max_sharpe_portfolio = optimal_portfolios['max_sharpe']
    
    # Calculate the efficient frontier
    efficient_risk, efficient_return = calculate_efficient_frontier(returns, cov_matrix, num_assets)
    
    # Calculate Capital Market Line
    risk_free = risk_free_rate/100
    cml_x = np.linspace(0, max(results[0]) * 1.2, 100)
    # Using the tangency portfolio (max Sharpe ratio) for the CML
    slope = (max_sharpe_portfolio['return'] - risk_free) / max_sharpe_portfolio['risk']
    cml_y = risk_free + slope * cml_x
    
    # Display results in columns
    col1, col2 = st.columns([1, 3])
    with col1:
        st.success(f"### Optimal Portfolios")
        
        # MPT Analysis
        st.markdown("### Key Portfolio Metrics")
        st.markdown(f"""
        - **Minimum Variance Portfolio:**
          - Return: `{min_var_portfolio['return']*100:.2f}%`
          - Risk: `{min_var_portfolio['risk']*100:.2f}%`
          - Weights: `Stock 1: {min_var_portfolio['weights'][0]*100:.1f}%, Stock 2: {min_var_portfolio['weights'][1]*100:.1f}%`
        
        - **Maximum Sharpe Ratio Portfolio:**
          - Return: `{max_sharpe_portfolio['return']*100:.2f}%`
          - Risk: `{max_sharpe_portfolio['risk']*100:.2f}%`
          - Sharpe Ratio: `{(max_sharpe_portfolio['return']-risk_free)/max_sharpe_portfolio['risk']:.2f}`
          - Weights: `Stock 1: {max_sharpe_portfolio['weights'][0]*100:.1f}%, Stock 2: {max_sharpe_portfolio['weights'][1]*100:.1f}%`
        """)
        
        # MPT Interpretation
        st.info(f"""
        ### Interpretation:
        
        The correlation between your assets is **{correlation:.2f}**, which affects diversification benefits.
        
        **Diversification Impact:** {
            "Excellent diversification benefits with negative correlation. This significantly reduces portfolio risk below what you'd expect from individual assets." if correlation < -0.3 else
            "Good diversification benefits with low correlation. This helps reduce portfolio risk." if correlation < 0.3 else
            "Moderate diversification benefits. Assets move somewhat independently." if correlation < 0.6 else
            "Limited diversification benefits with high correlation. These assets tend to move together."
        }
        
        **Key insight:** {
            "The efficient frontier shows the optimal risk-return tradeoff. The Maximum Sharpe Ratio portfolio represents the optimal mix for risk-adjusted returns." if correlation > -0.95 and correlation < 0.95 else
            "With nearly perfect correlation, diversification provides minimal benefit. Consider finding less correlated assets." if correlation > 0.95 else
            "With nearly perfect negative correlation, you can potentially create extremely low-risk portfolios through proper asset weighting."
        }
        """)

    with col2:
        # Generate portfolio visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot random portfolios
        scatter = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
        
        # Plot individual assets
        ax.scatter(vol, returns, marker='*', s=100, c='r', label='Individual Assets')
        
        # Plot efficient frontier
        ax.plot(efficient_risk, efficient_return, 'b--', linewidth=2, label='Efficient Frontier')
        
        # Plot minimum variance and tangency portfolios
        ax.scatter(min_var_portfolio['risk'], min_var_portfolio['return'], s=100, c='g', marker='o', label='Min Variance')
        ax.scatter(max_sharpe_portfolio['risk'], max_sharpe_portfolio['return'], s=100, c='y', marker='o', label='Max Sharpe Ratio')
        
        # Plot Capital Market Line
        ax.plot(cml_x, cml_y, 'r-', label='Capital Market Line')
        
        # Plot risk-free rate
        ax.scatter(0, risk_free, s=80, c='m', marker='s', label='Risk-Free Asset')
        
        # Add colorbar to show Sharpe ratio scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        ax.set_title("Modern Portfolio Theory Visualization", fontsize=14, fontweight='bold')
        ax.set_xlabel("Risk (Standard Deviation %)")
        ax.set_ylabel("Expected Return (%)")
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Convert to percentages for better visualization
        ax.set_xticklabels([f'{x:.0f}%' for x in ax.get_xticks()*100])
        ax.set_yticklabels([f'{y:.0f}%' for y in ax.get_yticks()*100])
        
        st.pyplot(fig)
        
        # MPT Confidence
        st.warning("""
        **📉 Portfolio Theory Considerations**
        
        Modern Portfolio Theory makes several assumptions that may not hold in all market conditions:
        
        - Returns are normally distributed
        - Correlations remain stable over time
        - Investors are rational and risk-averse
        - Past performance doesn't guarantee future results
        
        Remember that real-world markets often exhibit more complex behavior than this simplified model.
        """)

with tab2:
    st.markdown("""
    ## Modern Portfolio Theory: Mathematical Foundation
    
    ### What is Modern Portfolio Theory?
    
    Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952, is a framework for constructing investment portfolios that maximize expected return for a given level of risk, or minimize risk for a given level of expected return.
    
    ### Core Concepts
    
    **1. Risk and Return**
    
    MPT represents asset returns as normal distributions with:
    - Expected return (μ): The average anticipated return
    - Risk (σ): Standard deviation or volatility of returns
    
    **2. The Importance of Correlation**
    
    The crucial insight of MPT is that an asset's risk contribution to a portfolio depends on its correlation with other assets. This makes diversification powerful - by combining assets with less than perfect correlation, investors can reduce portfolio risk without sacrificing return.
    
    **3. Mathematical Framework**
    
    For a portfolio with N assets:
    
    Portfolio Expected Return:
    $$
    \\mu_\\Pi = \\sum_{i=1}^{N} W_i\\mu_i
    $$
    
    Portfolio Risk (Standard Deviation):
    $$
    \\sigma_\\Pi = \\sqrt{\\sum_{i=1}^{N}\\sum_{j=1}^{N}W_iW_j\\rho_{ij}\\sigma_i\\sigma_j}
    $$
    
    Where:
    - $W_i$ is the fraction of wealth invested in asset i
    - $\\mu_i$ is the expected return of asset i
    - $\\sigma_i$ is the standard deviation of asset i
    - $\\rho_{ij}$ is the correlation between assets i and j
    
    **4. Efficient Frontier**
    
    The efficient frontier represents portfolios that maximize return for a given level of risk. Portfolios on this curve are considered "efficient" - no other portfolio offers higher return for the same risk or lower risk for the same return.
    
    **5. Capital Market Line**
    
    When a risk-free asset is introduced, the efficient frontier becomes the tangent line from the risk-free rate to the efficient frontier. This line is called the Capital Market Line (CML).
    
    $$
    \\text{CML: } E(R_p) = R_f + \\frac{E(R_M) - R_f}{\\sigma_M}\\sigma_p
    $$
    
    Where:
    - $E(R_p)$ is the expected return of portfolio p
    - $R_f$ is the risk-free rate
    - $E(R_M)$ is the expected return of the market portfolio
    - $\\sigma_M$ is the standard deviation of the market portfolio
    - $\\sigma_p$ is the standard deviation of portfolio p
    
    **6. Market Portfolio**
    
    The point where the CML is tangent to the efficient frontier is called the Market Portfolio. According to MPT, all investors should hold a combination of the risk-free asset and the market portfolio, varying only in the proportion between them based on risk tolerance.
    
    **7. Sharpe Ratio**
    
    The slope of the CML is also known as the Sharpe ratio, which measures excess return per unit of risk:
    
    $$
    \\text{Sharpe Ratio} = \\frac{E(R_p) - R_f}{\\sigma_p}
    $$
    
    The higher the Sharpe ratio, the better the risk-adjusted performance.
    """)
    
    with st.expander("🔍 Hands-On Theoretical Exercise"):
        st.markdown("""
        **Calculate Portfolio Risk and Return:**
        
        1. Asset A: Expected Return = 5%, Risk = 15%
        2. Asset B: Expected Return = 10%, Risk = 25%
        3. Correlation between A and B = 0.3
        
        If we create a portfolio with 60% in Asset A and 40% in Asset B:
        
        **Portfolio Return** = 0.6 × 5% + 0.4 × 10% = 7%
        
        **Portfolio Risk** = $\\sqrt{(0.6^2 × 15%^2) + (0.4^2 × 25%^2) + (2 × 0.6 × 0.4 × 0.3 × 15% × 25%)}$ = 15.5%
        
        This is lower than the weighted average of individual risks (19%), demonstrating the benefit of diversification!
        """)

with tab3:
    st.markdown("""
    ## Welcome to the MPT Learning Tool!
    
    **What this tool does:**  
    This interactive calculator helps you visualize key concepts of Modern Portfolio Theory, including the efficient frontier, optimal portfolios, and the power of diversification.
    
    ### Quick Start Guide
    
    1. **Adjust Parameters** (Left Sidebar):
       - Set expected returns for each asset
       - Set risk (volatility) for each asset
       - Adjust correlation between assets
       - Set the risk-free rate
    
    2. **View Results** (Main Panel):
       - Portfolio combinations visualized in risk-return space
       - Efficient frontier showing optimal portfolios
       - Capital Market Line showing risk-free asset combinations
       - Key optimal portfolios highlighted
    
    3. **Try These Scenarios**:
       - 🎚️ Click "Set Low Correlation" to see how diversification reduces risk
       - ⚡ Click "Set High Correlation" to see limited diversification benefits
       - 😲 Click "Set Negative Correlation" to see powerful risk reduction
       - 🔄 Try adjusting the correlation slider to see how correlation affects the efficient frontier
       - 📊 Experiment with expected returns to see how they shift optimal portfolios
    
    ### Key Features to Explore
    - **Efficient Frontier**: The curved line showing optimal portfolios
    - **Maximum Sharpe Ratio Portfolio**: The portfolio with the best risk-adjusted return
    - **Minimum Variance Portfolio**: The portfolio with the lowest possible risk
    - **Capital Market Line**: Combinations of the risk-free asset and the optimal risky portfolio
    
    **Pro Tip:** Use the parameter sliders to create your own scenarios and test their implications!
    """)

    # Button row for scenario presets
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("Set Low Correlation", on_click=set_low_correlation_parameters)
    with col2:
        st.button("Set High Correlation", on_click=set_high_correlation_parameters)
    with col3:
        st.button("Set Negative Correlation", on_click=set_negative_correlation_parameters)
    with col4:
        st.button("Set Multiple Assets Scenario", on_click=set_multiple_assets_parameters)

# Tab 4: Practical Labs
with tab4:
    st.header("🔬 Practical MPT Labs")
    st.markdown("""
    Welcome to the **Practical MPT Labs** section! Each lab provides a real-world scenario or demonstration 
    to help you apply Modern Portfolio Theory concepts in a hands-on way.
    
    Use the **"Set Lab Parameters"** buttons to jump directly to recommended settings for each scenario.
    Experiment, take notes, and enjoy exploring how MPT works under different market conditions!
    """)

    # --- Additional Disclaimer ---
    st.warning("""
    **Disclaimer**:  
    The author does not endorse any particular investment strategy.
    This material is purely for educational and illustrative purposes.
    """)

    # A radio to choose one of the labs
    lab_choice = st.radio(
        "Select a lab to view:",
        ("Lab 1: The Power of Diversification",
         "Lab 2: Correlation and Portfolio Risk",
         "Lab 3: Finding the Optimal Portfolio",
         "Lab 4: Capital Allocation Line",
         "Lab 5: Risk-Return Tradeoffs"),
        index=0
    )

    # ---------------- Lab 1 ----------------
    if lab_choice == "Lab 1: The Power of Diversification":
        st.subheader("🔄 Lab 1: The Power of Diversification")
        st.markdown("""
        **Real-World Scenario:**  
        You're a portfolio manager with two potential investments: a technology stock with high expected return but also high volatility, 
        and a consumer staples stock with lower return but also lower risk. How should you combine them to achieve the best risk-adjusted return?

        ---
        **Beginner Explanation of What's Happening:**

        - **Diversification Benefit**: When you combine assets that don't move in perfect lockstep (correlation < 1.0),
          the portfolio's risk can be lower than the weighted average of individual risks.
        
        - **Risk Reduction**: The lower the correlation between assets, the greater the risk reduction from diversification.
          
        - **Efficient Combinations**: Some combinations of assets will be more efficient than others, offering better
          return for a given level of risk.

        **Learning Objective:**  
        - Understand how combining assets affects portfolio risk and return
        - Observe how correlation impacts diversification benefits
        - Identify the minimum variance and maximum Sharpe ratio portfolios

        ---
        **Suggested Steps**:
        1. Click "**Set Low Correlation**" to see a typical scenario with moderate diversification benefits.
        2. Observe how the efficient frontier curves away from the straight line connecting the individual assets.
        3. Note that the minimum variance portfolio has lower risk than either individual asset.
        4. Try reducing correlation further to see even greater risk reduction.
        5. Find the maximum Sharpe ratio portfolio on the efficient frontier.

        **💡 Reflection Questions:**  
        - Why does the efficient frontier curve bend more as correlation decreases?
        - What happens to the weights of the minimum variance portfolio as correlation changes?
        - How would your allocation strategy change in high vs. low correlation environments?
        """)

    # ---------------- Lab 2 ----------------
    elif lab_choice == "Lab 2: Correlation and Portfolio Risk":
        st.subheader("📉 Lab 2: Correlation and Portfolio Risk")
        st.markdown("""
        **Real-World Scenario:**  
        You're analyzing how different correlation scenarios affect portfolio performance. You have access to two assets with similar expected returns but different risk profiles. How does changing their correlation affect the shape of the efficient frontier and optimal allocations?

        **Learning Objective:**  
        - Examine the relationship between correlation and portfolio risk
        - Understand the extreme cases of perfect positive and negative correlation
        - Recognize how correlation affects the efficient frontier's shape

        ---
        **Suggested Steps**:
        1. Click "**Set High Correlation**" to see what happens with highly correlated assets.
        2. Note the limited diversification benefit (small bend in the efficient frontier).
        3. Click "**Set Negative Correlation**" to see powerful diversification effects.
        4. Observe how the efficient frontier bends dramatically with negative correlation.
        5. Try setting correlation to extreme values (-0.9 or +0.9) to see the limiting cases.

        **Key Insight**:  
        - With perfect positive correlation (ρ = 1), the efficient frontier becomes a straight line
        - With perfect negative correlation (ρ = -1), you can theoretically create a risk-free portfolio from risky assets
        - In real markets, most assets have positive correlations, but they often decrease during normal periods and increase during crises

        **💡 Reflection Questions:**  
        - Why might correlations between assets increase during market crises?
        - How much additional risk reduction do you get going from 0.5 correlation to 0 correlation versus from 0 to -0.5?
        - What types of assets typically have low or negative correlations with equities?
        """)

    # ---------------- Lab 3 ----------------
    elif lab_choice == "Lab 3: Finding the Optimal Portfolio":
        st.subheader("🎯 Lab 3: Finding the Optimal Portfolio")
        st.markdown("""
        **Real-World Scenario:**  
        You're advising a client with a moderate risk tolerance who wants to maximize return while controlling risk. 
        How do you identify the optimal portfolio for this investor? And how does the introduction of a risk-free asset change your recommendation?

        **Learning Objective:**  
        - Understand the difference between the minimum variance and maximum Sharpe ratio portfolios
        - Learn how the Capital Market Line works
        - Apply the concept of the optimal risky portfolio

        ---
        **Suggested Steps**:
        1. Reset parameters to default using the "Reset Parameters" button in the sidebar.
        2. Note the location of the minimum variance portfolio (green dot) and maximum Sharpe ratio portfolio (yellow dot).
        3. Observe the Capital Market Line (red) connecting the risk-free asset to the tangency portfolio.
        4. Try changing the risk-free rate and see how it affects the Capital Market Line and optimal portfolio.
        5. Consider different investor risk preferences and where they would be positioned on the Capital Market Line.

        **Why This Matters**:  
        - The maximum Sharpe ratio portfolio is the optimal risky portfolio for all investors
        - Investors can adjust risk by combining this portfolio with risk-free borrowing or lending
        - This separation theorem simplifies portfolio construction to a two-step process: find the optimal risky portfolio, then adjust allocation between risky and risk-free assets

        **💡 Reflection Questions:**  
        - Why is the maximum Sharpe ratio portfolio optimal for all investors regardless of risk tolerance?
        - How would a conservative investor use the Capital Market Line versus an aggressive investor?
        - What happens to the optimal portfolio weights as the risk-free rate changes?
        """)

    # ---------------- Lab 4 ----------------
    elif lab_choice == "Lab 4: Capital Allocation Line":
        st.subheader("💰 Lab 4: Capital Allocation Line and Risk Preferences")
        st.markdown("""
        **Real-World Scenario:**  
        You're a financial advisor with clients who have different risk tolerances. Some are conservative retirees, others are aggressive young professionals. How do you use MPT to create appropriate portfolios for each?

        **Learning Objective:**  
        - Understand how to use the Capital Allocation Line (CAL) to adjust risk exposure
        - Learn how to create portfolios that match specific risk preferences
        - Apply the concept of utility functions to portfolio selection

        ---
        **Suggested Steps**:
        1. Reset parameters to default values.
        2. Identify the maximum Sharpe ratio portfolio (tangency portfolio).
        3. Note that the Capital Market Line represents combinations of this portfolio and the risk-free asset.
        4. For a conservative investor, visualize a point on the CML closer to the risk-free asset.
        5. For an aggressive investor, visualize a point further out on the CML (potentially with leverage).

        **Practical Application**:  
        - Conservative investors might allocate 30% to the optimal risky portfolio and 70% to the risk-free asset
        - Moderate investors might allocate 70% to the optimal risky portfolio and 30% to the risk-free asset
        - Aggressive investors might allocate 100% or more (with leverage) to the optimal risky portfolio
        
        **The math behind this**:  
        If we call the weight of the risky portfolio w, then:
        - Expected return: E(Rp) = w × E(Rrisky) + (1-w) × Rf
        - Portfolio risk: σp = w × σrisky
        
        A risk-averse investor's utility function might look like:  
        U = E(Rp) - 0.5 × A × σp²  
        
        Where A is the investor's risk aversion coefficient.

        **💡 Reflection Questions:**  
        - How does an investor's time horizon typically affect their optimal position on the CAL?
        - What practical limitations exist when implementing a leveraged position (w > 100%)?
        - How might changing market conditions affect your recommendation to move along the CAL?
        """)

    # ---------------- Lab 5 ----------------
    else:  # lab_choice == "Lab 5: Risk-Return Tradeoffs"
        st.subheader("⚖️ Lab 5: Risk-Return Tradeoffs")
        st.markdown("""
        **Real-World Scenario:**  
        You're analyzing historical data for different asset classes (stocks, bonds, real estate, commodities) to determine their risk-return profiles and optimal allocations. How do you apply MPT principles to create a well-diversified multi-asset portfolio?

        **Learning Objective:**  
        - Understand risk-return tradeoffs across different asset classes
        - Apply MPT to multi-asset allocation decisions
        - Recognize the real-world limitations of MPT

        ---
        **Suggested Steps**:
        1. Click "**Set Multiple Assets Scenario**" to simulate a multi-asset environment.
        2. Observe how adding more assets potentially improves the efficient frontier.
        3. Note how different assets contribute to the optimal portfolio based on their risk, return, and correlation characteristics.
        4. Consider how different economic scenarios might affect these relationships.

        **Key Considerations**:  
        - Historical data has limitations - past relationships may not persist
        - Correlations tend to increase during market stress when diversification is most needed
        - Asset return distributions often have "fat tails" not captured by normal distributions
        - Practical implementation requires consideration of liquidity, taxes, and transaction costs

        **Real-World Extensions of MPT**:
        - **Black-Litterman Model**: Combines market equilibrium with investor views
        - **Post-Modern Portfolio Theory**: Uses downside risk measures instead of variance
        - **Factor Models**: Focus on underlying risk factors rather than asset correlations
        - **Robust Optimization**: Accounts for uncertainty in parameter estimates

        **💡 Reflection Questions:**  
        - How might the efficient frontier change during different economic regimes?
        - What non-quantitative factors should supplement MPT when making investment decisions?
        - How frequently should optimal portfolios be rebalanced in practice?
        """)

with tab5:
    st.header("🧮 Playground: Interactive MPT Learning")
    
    st.markdown("""
    Welcome to the MPT Playground! This section offers interactive activities to deepen your understanding 
    of Modern Portfolio Theory concepts through hands-on experimentation.
    """)
    
    activity = st.selectbox(
        "Choose an activity:",
        [
            "Portfolio Constructor",
            "Correlation Explorer",
            "Efficient Frontier Builder",
            "MPT Challenge Quiz"
        ]
    )
    
    if activity == "Portfolio Constructor":
        st.subheader("Portfolio Constructor")
        st.markdown("""
        Build your own portfolio by allocating between assets and see how it performs in risk-return space.
        
        **Instructions:**
        1. Set the weights for each asset
        2. See where your portfolio falls on the efficient frontier
        3. Compare its performance to optimal portfolios
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Portfolio Allocation")
            
            # User inputs their allocation
            asset1_weight = st.slider("Asset 1 Weight (%)", 0, 100, 50)
            asset2_weight = 100 - asset1_weight
            
            st.info(f"Asset 2 Weight: {asset2_weight}%")
            
            weights = np.array([asset1_weight/100, asset2_weight/100])
            
            # Current parameters
            st.markdown("### Current Parameters")
            st.markdown(f"""
            - Asset 1: Return = {stock1_return}%, Risk = {stock1_risk}%
            - Asset 2: Return = {stock2_return}%, Risk = {stock2_risk}%
            - Correlation: {correlation}
            - Risk-free rate: {risk_free_rate}%
            """)
            
            calculate_button = st.button("Calculate Portfolio Performance")
        
        with col2:
            st.markdown("### Portfolio Analysis")
            
            if calculate_button:
                # Calculate performance of user portfolio
                returns = np.array([stock1_return/100, stock2_return/100])
                vol = np.array([stock1_risk/100, stock2_risk/100])
                corr_matrix = np.array([[1, correlation], [correlation, 1]])
                cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)
                
                user_return, user_risk = calculate_portfolio_performance(weights, returns, cov_matrix)
                user_sharpe = (user_return - risk_free_rate/100) / user_risk
                
                # Find optimal portfolios for comparison
                optimal_portfolios = find_optimal_portfolios(returns, cov_matrix, 2)
                min_var_portfolio = optimal_portfolios['min_var']
                max_sharpe_portfolio = optimal_portfolios['max_sharpe']
                
                # Calculate the efficient frontier
                efficient_risk, efficient_return = calculate_efficient_frontier(returns, cov_matrix, 2)
                
                # Display user portfolio performance
                st.markdown(f"""
                ### Your Portfolio Performance
                
                - **Expected Return:** {user_return*100:.2f}%
                - **Risk (Volatility):** {user_risk*100:.2f}%
                - **Sharpe Ratio:** {user_sharpe:.2f}
                
                **Comparison to Optimal Portfolios:**
                
                - **vs. Minimum Variance:** 
                  - Return: {(user_return - min_var_portfolio['return'])*100:+.2f}%
                  - Risk: {(user_risk - min_var_portfolio['risk'])*100:+.2f}%
                
                - **vs. Maximum Sharpe Ratio:** 
                  - Return: {(user_return - max_sharpe_portfolio['return'])*100:+.2f}%
                  - Risk: {(user_risk - max_sharpe_portfolio['risk'])*100:+.2f}%
                  - Sharpe: {user_sharpe - (max_sharpe_portfolio['return'] - risk_free_rate/100) / max_sharpe_portfolio['risk']:+.2f}
                """)
                
                # Evaluation of portfolio efficiency
                is_efficient = False
                for i in range(len(efficient_risk)):
                    if abs(user_risk - efficient_risk[i]) < 0.001 and user_return >= efficient_return[i] - 0.001:
                        is_efficient = True
                        break
                
                if is_efficient:
                    st.success("✅ Your portfolio appears to be on or very near the efficient frontier!")
                else:
                    st.warning("⚠️ Your portfolio is not on the efficient frontier. You could achieve a better return for the same risk.")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot the efficient frontier
                ax.plot(efficient_risk, efficient_return, 'b--', linewidth=2, label='Efficient Frontier')
                
                # Plot the individual assets
                ax.scatter(vol, returns, marker='*', s=100, c='r', label='Individual Assets')
                
                # Plot min variance and max Sharpe portfolios
                ax.scatter(min_var_portfolio['risk'], min_var_portfolio['return'], s=100, c='g', marker='o', label='Min Variance')
                ax.scatter(max_sharpe_portfolio['risk'], max_sharpe_portfolio['return'], s=100, c='y', marker='o', label='Max Sharpe')
                
                # Plot user portfolio
                ax.scatter(user_risk, user_return, s=150, c='purple', marker='X', label='Your Portfolio')
                
                # Format plot
                ax.set_title("Your Portfolio vs. Efficient Frontier")
                ax.set_xlabel("Risk (Standard Deviation)")
                ax.set_ylabel("Expected Return")
                ax.grid(alpha=0.3)
                ax.legend()
                
                # Convert to percentages for better visualization
                ax.set_xticklabels([f'{x:.0f}%' for x in ax.get_xticks()*100])
                ax.set_yticklabels([f'{y:.0f}%' for y in ax.get_yticks()*100])
                
                st.pyplot(fig)
            else:
                st.info("Click 'Calculate Portfolio Performance' to see analysis")
    
    elif activity == "Correlation Explorer":
        st.subheader("Correlation Explorer")
        st.markdown("""
        Visualize how changing correlation between assets affects portfolio risk and the efficient frontier.
        
        **Instructions:**
        1. Use the slider to change correlation between assets
        2. Observe how the efficient frontier shape changes
        3. See how minimum variance portfolio weights change
        """)
        
        # Correlation slider
        explorer_correlation = st.slider("Correlation between Assets", -1.0, 1.0, 0.0, 0.1)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Effect on Diversification")
            
            # Calculate the impact of correlation on a 50/50 portfolio
            returns = np.array([stock1_return/100, stock2_return/100])
            vol = np.array([stock1_risk/100, stock2_risk/100])
            
            # Calculate risk for different correlations
            correlations = [-0.9, -0.5, 0.0, 0.5, 0.9, explorer_correlation]
            portfolio_risks = []
            
            for corr in correlations:
                corr_matrix = np.array([[1, corr], [corr, 1]])
                cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)
                _, risk = calculate_portfolio_performance(np.array([0.5, 0.5]), returns, cov_matrix)
                portfolio_risks.append(risk * 100)  # Convert to percentage
            
            # Calculate minimum variance portfolio weights
            corr_matrix = np.array([[1, explorer_correlation], [explorer_correlation, 1]])
            cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)
            
            min_var_optimal = find_optimal_portfolios(returns, cov_matrix, 2)['min_var']
            
            # Calculate weighted average risk
            weighted_avg_risk = 0.5 * stock1_risk + 0.5 * stock2_risk
            
            # Display impact
            st.markdown(f"""
            **50/50 Portfolio Risk:**
            
            - With ρ = -0.9: {portfolio_risks[0]:.2f}%
            - With ρ = -0.5: {portfolio_risks[1]:.2f}%
            - With ρ = 0.0: {portfolio_risks[2]:.2f}%
            - With ρ = 0.5: {portfolio_risks[3]:.2f}%
            - With ρ = 0.9: {portfolio_risks[4]:.2f}%
            - With ρ = {explorer_correlation}: {portfolio_risks[5]:.2f}%
            
            **Current Risk Reduction:**
            
            50/50 Portfolio Risk: {portfolio_risks[5]:.2f}%
            Weighted Avg Risk: {weighted_avg_risk:.2f}%
            Risk Reduction: {weighted_avg_risk - portfolio_risks[5]:.2f}%
            
            **Minimum Variance Portfolio:**
            
            Asset 1 Weight: {min_var_optimal['weights'][0]*100:.1f}%
            Asset 2 Weight: {min_var_optimal['weights'][1]*100:.1f}%
            Portfolio Risk: {min_var_optimal['risk']*100:.2f}%
            """)
            
            # Correlation interpretation
            if explorer_correlation < -0.5:
                st.success("Strong negative correlation provides exceptional diversification benefits.")
            elif explorer_correlation < 0:
                st.success("Negative correlation provides very good diversification benefits.")
            elif explorer_correlation < 0.3:
                st.info("Low positive correlation provides good diversification benefits.")
            elif explorer_correlation < 0.7:
                st.info("Moderate correlation provides some diversification benefits.")
            else:
                st.warning("High correlation limits diversification benefits.")
        
        with col2:
            # Generate efficient frontiers for different correlations
            fig, ax = plt.subplots(figsize=(10, 6))
            
            correlations_to_plot = [-0.5, 0.0, 0.5, explorer_correlation]
            correlation_colors = ['green', 'blue', 'orange', 'red']
            correlation_labels = ['ρ = -0.5', 'ρ = 0.0', 'ρ = 0.5', f'ρ = {explorer_correlation}']
            
            # Plot individual assets
            ax.scatter([stock1_risk/100, stock2_risk/100], [stock1_return/100, stock2_return/100], 
                      marker='*', s=100, c='black', label='Individual Assets')
            
            # Plot efficient frontiers for different correlations
            for i, corr in enumerate(correlations_to_plot):
                # Skip duplicates (if explorer_correlation equals one of the predefined values)
                if i < 3 and corr == explorer_correlation:
                    continue
                
                corr_matrix = np.array([[1, corr], [corr, 1]])
                cov_matrix = np.diag(vol) @ corr_matrix @ np.diag(vol)
                
                efficient_risk, efficient_return = calculate_efficient_frontier(returns, cov_matrix, 2)
                
                ax.plot(efficient_risk, efficient_return, linewidth=2, color=correlation_colors[i], 
                       label=correlation_labels[i])
                
                # Plot minimum variance portfolio
                min_var = find_optimal_portfolios(returns, cov_matrix, 2)['min_var']
                ax.scatter(min_var['risk'], min_var['return'], s=100, color=correlation_colors[i], marker='o')
            
            ax.set_title("Effect of Correlation on the Efficient Frontier")
            ax.set_xlabel("Risk (Standard Deviation)")
            ax.set_ylabel("Expected Return")
            ax.grid(alpha=0.3)
            ax.legend()
            
            # Convert to percentages for better visualization
            ax.set_xticklabels([f'{x:.0f}%' for x in ax.get_xticks()*100])
            ax.set_yticklabels([f'{y:.0f}%' for y in ax.get_yticks()*100])
            
            st.pyplot(fig)
            
            st.markdown("""
            **Key Observations:**
            
            1. As correlation decreases, the efficient frontier bends more to the left, indicating greater diversification benefit
            2. With negative correlation, the minimum variance portfolio can have lower risk than either individual asset
            3. With high positive correlation, the efficient frontier approaches a straight line
            4. The minimum variance portfolio weights shift toward the lower-risk asset as correlation increases
            """)
    
    elif activity == "Efficient Frontier Builder":
        st.subheader("Efficient Frontier Builder")
        st.markdown("""
        Experiment with building efficient frontiers using different asset characteristics.
        
        **Instructions:**
        1. Set parameters for multiple assets
        2. Generate the efficient frontier
        3. Explore how changing parameters affects the optimal portfolios
        """)
        
        num_assets = st.radio("Number of assets to include:", [2, 3, 4], horizontal=True)
        
        # Create columns for asset inputs
        cols = st.columns(num_assets)
        
        # Arrays to store asset parameters
        returns_array = np.zeros(num_assets)
        risk_array = np.zeros(num_assets)
        
        # Get user inputs for each asset
        for i in range(num_assets):
            with cols[i]:
                st.markdown(f"### Asset {i+1}")
                returns_array[i] = st.slider(f"Expected Return (%)", 0.0, 20.0, 5.0 + i*3.0) / 100
                risk_array[i] = st.slider(f"Risk (%)", 5.0, 50.0, 10.0 + i*5.0) / 100
        
        # Correlation matrix input
        st.markdown("### Correlation Matrix")
        
        # Create a default correlation matrix (0.3 between all pairs)
        default_corr = np.ones((num_assets, num_assets))
        for i in range(num_assets):
            for j in range(num_assets):
                if i != j:
                    default_corr[i, j] = 0.3
        
        # Let user adjust correlations
        corr_matrix = np.ones((num_assets, num_assets))
        
        if num_assets > 2:
            st.markdown("Enter correlations between asset pairs:")
            
            # Create rows for correlation inputs
            for i in range(num_assets):
                cols = st.columns(num_assets)
                for j in range(num_assets):
                    with cols[j]:
                        if i == j:
                            st.markdown(f"Asset {i+1}")
                            corr_matrix[i, j] = 1.0
                        elif j > i:
                            corr_matrix[i, j] = st.slider(
                                f"ρ Asset {i+1}-{j+1}",
                                -1.0, 1.0, 0.3, 0.1,
                                key=f"corr_{i}_{j}"
                            )
                            corr_matrix[j, i] = corr_matrix[i, j]  # Make symmetric
                        else:
                            # Display the value from the other side of the matrix
                            st.markdown(f"ρ: {corr_matrix[i, j]:.1f}")
        else:
            # Simplified for 2 assets
            corr_matrix[0, 1] = corr_matrix[1, 0] = st.slider("Correlation between assets", -1.0, 1.0, 0.3, 0.1)
            
        # Button to generate the efficient frontier
        if st.button("Generate Efficient Frontier"):
            # Create covariance matrix
            cov_matrix = np.diag(risk_array) @ corr_matrix @ np.diag(risk_array)
            
            # Generate random portfolios
            results, weights_record = generate_random_portfolios(1000, returns_array, cov_matrix, num_assets)
            
            # Find optimal portfolios
            optimal_portfolios = find_optimal_portfolios(returns_array, cov_matrix, num_assets)
            min_var_portfolio = optimal_portfolios['min_var']
            max_sharpe_portfolio = optimal_portfolios['max_sharpe']
            
            # Calculate the efficient frontier
            efficient_risk, efficient_return = calculate_efficient_frontier(returns_array, cov_matrix, num_assets)
            
            # Display efficient frontier
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot random portfolios
            scatter = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
            
            # Plot individual assets
            for i in range(num_assets):
                ax.scatter(risk_array[i], returns_array[i], marker='*', s=100, c='r')
                ax.annotate(f"Asset {i+1}", (risk_array[i], returns_array[i]), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            # Plot efficient frontier
            ax.plot(efficient_risk, efficient_return, 'b--', linewidth=2, label='Efficient Frontier')
            
            # Plot minimum variance and tangency portfolios
            ax.scatter(min_var_portfolio['risk'], min_var_portfolio['return'], s=100, c='g', marker='o', label='Min Variance')
            ax.scatter(max_sharpe_portfolio['risk'], max_sharpe_portfolio['return'], s=100, c='y', marker='o', label='Max Sharpe Ratio')
            
            # Add colorbar to show Sharpe ratio scale
            cbar = plt.colorbar(scatter)
            cbar.set_label('Sharpe Ratio')
            
            ax.set_title("Efficient Frontier with Optimal Portfolios")
            ax.set_xlabel("Risk (Standard Deviation)")
            ax.set_ylabel("Expected Return")
            ax.grid(alpha=0.3)
            ax.legend()
            
            # Convert to percentages for better visualization
            ax.set_xticklabels([f'{x:.0f}%' for x in ax.get_xticks()*100])
            ax.set_yticklabels([f'{y:.0f}%' for y in ax.get_yticks()*100])
            
            st.pyplot(fig)
            
            # Display optimal portfolio weights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Minimum Variance Portfolio")
                
                st.markdown(f"""
                **Expected Return:** {min_var_portfolio['return']*100:.2f}%
                **Risk:** {min_var_portfolio['risk']*100:.2f}%
                **Sharpe Ratio:** {(min_var_portfolio['return'] - risk_free_rate/100) / min_var_portfolio['risk']:.2f}
                
                **Asset Weights:**
                """)
                
                for i in range(num_assets):
                    st.markdown(f"- Asset {i+1}: {min_var_portfolio['weights'][i]*100:.1f}%")
            
            with col2:
                st.markdown("### Maximum Sharpe Ratio Portfolio")
                
                st.markdown(f"""
                **Expected Return:** {max_sharpe_portfolio['return']*100:.2f}%
                **Risk:** {max_sharpe_portfolio['risk']*100:.2f}%
                **Sharpe Ratio:** {(max_sharpe_portfolio['return'] - risk_free_rate/100) / max_sharpe_portfolio['risk']:.2f}
                
                **Asset Weights:**
                """)
                
                for i in range(num_assets):
                    st.markdown(f"- Asset {i+1}: {max_sharpe_portfolio['weights'][i]*100:.1f}%")
    
    else:  # MPT Challenge Quiz
        st.subheader("MPT Challenge Quiz")
        st.markdown("""
        Test your knowledge of Modern Portfolio Theory with this quiz.
        """)
        
        # Initialize session state if needed
        if 'quiz_score' not in st.session_state:
            st.session_state.quiz_score = 0
            st.session_state.questions_answered = 0
            st.session_state.current_question = 0
        
        # Quiz questions
        questions = [
            {
                "question": "Which of the following best describes Modern Portfolio Theory's key insight?",
                "options": [
                    "Assets should be selected based solely on their individual returns", 
                    "Risk is best measured by beta relative to the market", 
                    "Diversification can reduce portfolio risk without sacrificing expected return", 
                    "Markets are always efficient and cannot be beaten"
                ],
                "correct": 2,
                "explanation": "The key insight of MPT is that by combining assets with imperfect correlation, investors can reduce portfolio risk without sacrificing expected return."
            },
            {
                "question": "According to MPT, what happens to portfolio risk as correlation between assets decreases?",
                "options": [
                    "Risk increases", 
                    "Risk decreases", 
                    "Risk remains unchanged", 
                    "Risk becomes unpredictable"
                ],
                "correct": 1,
                "explanation": "As correlation between assets decreases, portfolio risk decreases due to greater diversification benefits."
            },
            {
                "question": "What is the efficient frontier in Modern Portfolio Theory?",
                "options": [
                    "The line connecting the risk-free asset to the market portfolio", 
                    "The set of portfolios with the highest return for a given level of risk", 
                    "The portfolio with the highest Sharpe ratio", 
                    "The boundary between acceptable and unacceptable investments"
                ],
                "correct": 1,
                "explanation": "The efficient frontier represents the set of portfolios that offer the highest expected return for a given level of risk, or equivalently, the lowest risk for a given level of expected return."
            },
            {
                "question": "What is the Capital Market Line?",
                "options": [
                    "The tangent line from the risk-free asset to the efficient frontier", 
                    "The line representing all possible portfolio combinations", 
                    "The boundary between bull and bear markets", 
                    "The relationship between GDP and market returns"
                ],
                "correct": 0,
                "explanation": "The Capital Market Line is the tangent line from the risk-free asset to the efficient frontier. It represents combinations of the risk-free asset and the optimal risky portfolio."
            },
            {
                "question": "Which portfolio on the efficient frontier does the Capital Market Line touch?",
                "options": [
                    "The minimum variance portfolio", 
                    "The maximum return portfolio", 
                    "The maximum Sharpe ratio portfolio", 
                    "The market portfolio"
                ],
                "correct": 2,
                "explanation": "The Capital Market Line touches the efficient frontier at the maximum Sharpe ratio portfolio, which is the tangency portfolio that offers the highest excess return per unit of risk."
            },
            {
                "question": "If two assets have a correlation of -1, what can be achieved by combining them?",
                "options": [
                    "Maximum possible return", 
                    "A risk-free portfolio", 
                    "The market return", 
                    "Zero correlation with the market"
                ],
                "correct": 1,
                "explanation": "With a perfect negative correlation (ρ = -1), it's theoretically possible to create a portfolio with zero risk (a risk-free portfolio) by properly weighting the two assets."
            },
            {
                "question": "According to Markowitz, how should portfolio A be compared to portfolio B?",
                "options": [
                    "By return only", 
                    "By risk only", 
                    "By both risk and return", 
                    "By Sharpe ratio only"
                ],
                "correct": 2,
                "explanation": "Markowitz's framework compares portfolios using both risk (standard deviation) and expected return. Portfolio A is at least as good as B if it has higher return and lower risk."
            },
            {
                "question": "What does the Sharpe ratio measure?",
                "options": [
                    "Total return", 
                    "Total risk", 
                    "Excess return per unit of risk", 
                    "Market correlation"
                ],
                "correct": 2,
                "explanation": "The Sharpe ratio measures excess return (portfolio return minus risk-free rate) per unit of risk (standard deviation), effectively quantifying risk-adjusted performance."
            },
            {
                "question": "According to MPT, which of these portfolios should rational investors hold?",
                "options": [
                    "Only the minimum variance portfolio", 
                    "Only the maximum return portfolio", 
                    "Any portfolio on the efficient frontier", 
                    "A combination of the risk-free asset and the tangency portfolio"
                ],
                "correct": 3,
                "explanation": "According to MPT, all rational investors should hold a combination of the risk-free asset and the tangency portfolio (maximum Sharpe ratio portfolio), adjusting the proportions based on their risk preferences."
            },
            {
                "question": "Which of the following is NOT an assumption of traditional MPT?",
                "options": [
                    "Returns follow a normal distribution", 
                    "Investors are rational and risk-averse", 
                    "Markets experience frequent crashes", 
                    "Investors care only about risk and return"
                ],
                "correct": 2,
                "explanation": "Traditional MPT does not assume that markets experience frequent crashes. Rather, it assumes normally distributed returns, which underestimates the likelihood of extreme events compared to what we observe in real markets."
            }
        ]
        
        # Display current question
        if st.session_state.current_question < len(questions):
            current_q = questions[st.session_state.current_question]
            
            st.markdown(f"**Question {st.session_state.current_question + 1} of {len(questions)}:**")
            st.markdown(f"### {current_q['question']}")
            
            # Display options
            user_answer = st.radio("Select your answer:", current_q['options'], key=f"q{st.session_state.current_question}")
            selected_index = current_q['options'].index(user_answer)
            
            if st.button("Submit Answer", key=f"submit_{st.session_state.current_question}"):
                if selected_index == current_q['correct']:
                    st.success("✅ Correct!")
                    st.session_state.quiz_score += 1
                else:
                    st.error("❌ Incorrect")
                    
                st.info(f"**Explanation:** {current_q['explanation']}")
                st.session_state.questions_answered += 1
                
                if st.button("Next Question", key=f"next_{st.session_state.current_question}"):
                    st.session_state.current_question += 1
                    st.experimental_rerun()
            
        else:
            # Quiz completed
            score_percentage = (st.session_state.quiz_score / len(questions)) * 100
            
            st.markdown(f"### Quiz Complete!")
            st.markdown(f"You scored: **{st.session_state.quiz_score}/{len(questions)}** ({score_percentage:.1f}%)")
            
            if score_percentage >= 80:
                st.success("🏆 Excellent! You have a strong understanding of Modern Portfolio Theory.")
            elif score_percentage >= 60:
                st.info("👍 Good job! You have a solid foundation in MPT concepts, with room to refine your understanding.")
            else:
                st.warning("📚 Keep learning! Review the MPT theory and practice more with the interactive tools.")
            
            if st.button("Restart Quiz"):
                st.session_state.quiz_score = 0
                st.session_state.questions_answered = 0
                st.session_state.current_question = 0
                st.experimental_rerun()

# Modern UI-style disclaimer (Bootstrap-like "alert-danger")
st.markdown("""
<div style="
    background-color: #f8d7da; 
    color: #721c24; 
    padding: 20px; 
    border-radius: 8px; 
    margin-bottom: 20px;
">
  <h4 style="margin-top: 0;">
    <strong>IMPORTANT DISCLAIMER</strong>
  </h4>
  <ul style="list-style-type: disc; padding-left: 1.5em;">
    <li>Modern Portfolio Theory makes <em>simplifying assumptions</em> that may not hold in real markets, including 
      normally distributed returns and stable correlations.</li>
    <li>Even the most sophisticated portfolio optimization techniques often <strong>fail to outperform</strong> a simple buy-and-hold strategy 
      in a diversified index over long time periods.</li>
    <li>Nobel laureate Harry Markowitz himself has acknowledged the theory's limitations in practical implementation.</li>
    <li>The reality is that <strong>markets are complex adaptive systems</strong> whose behavior may not be fully captured by mathematical models.</li>
    <li>Academic research suggests investors should focus on <strong>asset allocation, costs, and behavioral discipline</strong> 
      rather than sophisticated optimization techniques.</li>
    <li>This material is <strong>purely educational</strong>. The author does <strong>not</strong> recommend any particular
      investment strategy or approach.</li>
  </ul>
</div>
""", unsafe_allow_html=True)