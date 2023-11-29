import yfinance as yf
import numpy as np
import pandas as pd


def calculate_weights(stock_dict):
    """
    Calculates the weights for a given stock dictionary
    :param dict stock_dict: A dictionary in the form {'Ticker' : weight, 'Ticker' : weight, ...}
    :return: list of tickers, ndarray[Any, dtype] of weights
    """
    total_investment = sum(stock_dict.values())
    weights = np.array([amount / total_investment for amount in stock_dict.values()])
    tickers = list(stock_dict.keys())
    return tickers, weights


def download_stock_data(tickers, start_date, end_date):
    """
    Downloads data for a list of stock ticker strings
    :param list tickers: list of stock tickers to gather data for
    :param str start_date: start date for download in form 'YYYY-MM-DD'
    :param str end_date: end date for download in form 'YYYY-MM-DD'
    :return: pandas Dataframe with stock data
    """
    return yf.download(tickers, start_date, end_date, progress=False)


def mean_variance_optimization(tickers, start_date, end_date, max_volatility, expected_returns=None, min_weight=0.01, max_weight=0.35, simulations=10000):
    """
    Performs enhanced mean-variance optimization with weight constraints
    :param list tickers: list of stock tickers to optimize weights for
    :param string start_date: start date for analysis in form 'YYYY-MM-DD'
    :param str end_date: end date for analysis in form 'YYYY-MM-DD'
    :param float max_volatility: maximum annualized volatility
    :param float expected_returns: Optional input for expected returns of a stock
    :param float min_weight: minimum weight for each stock ticker
    :param float max_weight: maximum weight for each stock ticker
    :param int simulations: number of Monte Carlo simulations
    :return: optimal weights for each ticker
    """
    data = download_stock_data(tickers, start_date, end_date)['Adj Close']
    daily_returns = data.pct_change().dropna()

    if expected_returns is None:
        # Calculate expected returns from historical data
        expected_returns = daily_returns.mean() * 252

    covariance = daily_returns.cov()

    simulation_results = np.zeros((4, simulations))
    weights_record = np.zeros((len(tickers), simulations))

    for i in range(simulations):
        simulated_weights = np.random.uniform(min_weight, max_weight, len(tickers))
        simulated_weights /= np.sum(simulated_weights)  # Normalize to sum to 1

        weights_record[:, i] = simulated_weights

        annual_return = np.sum(simulated_weights * expected_returns)
        annual_stddev = np.sqrt(np.dot(simulated_weights.T, np.dot(covariance, simulated_weights))) * np.sqrt(252)
        mean_variance = annual_return / annual_stddev

        simulation_results[:, i] = [annual_return, annual_stddev, mean_variance, i]

    columns = ['Annualized Return', 'Annualized Volatility', 'Mean-Variance', 'Simulation Index']
    simulation_results = pd.DataFrame(simulation_results.T, columns=columns)
    filtered_results = simulation_results[simulation_results['Annualized Volatility'] <= max_volatility]
    optimal_mean_variance_idx = filtered_results['Mean-Variance'].idxmax()

    return weights_record[:, optimal_mean_variance_idx]

