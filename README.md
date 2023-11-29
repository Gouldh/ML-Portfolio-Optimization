# Stock Portfolio Machine Learning Optimization

## Project Overview
This project focuses on optimizing stock portfolios using various financial theories and machine learning models. It includes modules for factor analysis, mean-variance optimization, machine learning strategies for stock prediction, the Black-Litterman model for adjusting portfolio weights based on machine learning predictions, and portfolio statistics calculations. While rooted in the concepts established by my [Portfolio Analysis Suite](https://github.com/Gouldh/Portfolio-Analysis-Suite) project, this project stands on its own, offering unique functionalities and perspectives that distinguish it from its predecessor.

## Features
- **Mean-Variance Optimization**: Calculates the optimal asset allocation by balancing expected return against risk, subject to constraints on individual asset weights.
- **Machine Learning Strategies**: Employs various machine learning algorithms to forecast stock returns. It enhances the decision-making process by providing predictive insights based on historical data.
- **Black-Litterman Model**: Integrates market equilibrium and machine learning predictions to adjust expected returns. This model refines the portfolio optimization process by incorporating predictions into objective market data.
- **Portfolio Statistics**: Provides essential metrics such as the Sharpe Ratio, Sortino Ratio, and Information Ratio to evaluate the performance and risk-adjusted returns of investment portfolios.
- **Factor Analysis** (WIP): Examines how Fama-French factors like market risk, size, and value impact portfolio returns. This analysis is beneficial in understanding the driving forces behind portfolio performance.

## Libraries
- `pandas`: Essential for data manipulation and transformation. It is used to handle dataframes, perform data cleaning, and prepare datasets for analysis and visualization.
- `yfinance`: Primary source for downloading stock and market data. It is used across various modules to fetch historical stock prices and other financial data for analysis.
- `numpy`: Utilized for array and numerical computations. It is particularly used in the mean-variance optimization process for handling arrays and performing mathematical operations.
- `matplotlib` and `seaborn`: Both crucial for data visualization. While matplotlib is used for plotting graphs and charts, seaborn enhances these visualizations with more attractive and informative statistical graphics.
- `sklearn` (scikit-learn): Deployed for implementing and training machine learning models, such as Linear Regression, Random Forest, and Gradient Boosting, in the machine learning strategies module. It is also used for splitting data into training and test sets and evaluating model performance.
- `statsmodels`: Utilized for conducting statistical tests and models, particularly in factor analysis. It is used for running regression analyses to assess the impact of different financial factors on stock returns.


## Installation
To use the Portfolio Analysis Suite, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gouldh/ML-Portfolio-Optimization.git
   ```
2. Navigate to the repository's directory
   ```bash
   cd ML-Portfolio-Optimization
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Here's a quick start guide to using the modules in this project:

1. **Factor Analysis**:
   ```python
   import factor_analysis as fa
   factor_data = fa.download_factor_data('start_date', 'end_date')
   ```
2. **Mean-Variance Optimization**:
   ```python
   import mean_variance_optimization as mv
   optimized_weights = mv.mean_variance_optimization(tickers, 'start_date', 'end_date', max_volatility)
   ```
3. **Machine Learning Strategies**:
   ```python
   import machine_learning_strategies as mls
   predicted_return, confidence = mls.generate_investor_views('ticker', 'start_date', 'end_date')
   ```
4. **Black-Litterman Model**:
   ```python
   import black_litterman_model as bl
   adjusted_returns = bl.black_litterman_adjustment(market_returns, investor_views, view_confidences, historical_data)
   ```
5. **Portfolio Statistics**:
   ```python
   import portfolio_statistics as ps
   sharpe_ratio = ps.sharpe_ratio(returns, risk_free_rate)
   ```

## Sample Output
Below is an example of the output produced by running the code with sample input parameters. The table shows the allocation percentages for each stock in the original Mean-Variance Optimization, as well as the Machine Learning-enhanced Mean-Variance Optimization. The Chart shows the performance of the three portfolios against the market representation (SPY) and provices metrics for each portfolio's performance.

```plaintext
     Original MV Optimization ML MV Optimization
AAPL   16.67%          23.69%             25.00%
JNJ    13.33%           1.44%              8.97%
PG      8.89%           9.17%             15.00%
JPM    14.44%          22.96%              3.82%
XOM     7.78%           6.19%              3.89%
MMM     6.67%           5.58%              1.96%
SO      5.56%           2.05%              6.86%
VZ      6.67%          14.31%             22.06%
NKE    11.11%          13.36%              5.52%
DD      8.89%           1.24%              6.92%
```

![Example Output](https://github.com/Gouldh/ML-Portfolio-Optimization/blob/main/Example%20Code%20Output.png) 

## License
This project is open-sourced under the MIT License. For more information, please refer to the `LICENSE` file.

**Author**: Hunter Gould         
**Date**: 11/29/2023
