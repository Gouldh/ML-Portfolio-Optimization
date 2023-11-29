import numpy as np


def sharpe_ratio(returns, risk_free_rate):
    """
    Calculates the sharpe ratio for a set of returns
    :param returns: numpy array or pandas Series of returns for the investment
    :param float risk_free_rate: the risk-free rate of return, typically the yield on government bonds.
    :return: float, the calculated Sharpe Ratio
    """
    excess_returns = returns - (risk_free_rate / 252)
    annualized_excess_return = np.mean(excess_returns) * 252  # Assuming daily returns and 252 trading days in a year
    annualized_std_dev = np.std(excess_returns) * np.sqrt(252)
    sharpe = annualized_excess_return / annualized_std_dev
    return sharpe


def sortino_ratio(returns, risk_free_rate):
    """
    Calculates the Sortino Ratio for a set of investment returns
    :param returns: numpy array or pandas Series of returns for the investment
    :param risk_free_rate: he risk-free rate of return, typically the yield on government bonds.
    :return: float, the calculated Sortino Ratio
    """
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = np.minimum(excess_returns, 0)
    annualized_excess_return = np.mean(excess_returns) * 252
    annualized_downside_std_dev = np.std(downside_returns) * np.sqrt(252)
    sortino = annualized_excess_return / annualized_downside_std_dev
    return sortino


def information_ratio(returns, benchmark_returns):
    """
    Calculates the Information Ratio for a set of investment returns against a benchmark
    :param returns: numpy array or pandas Series of returns for the portfolio
    :param benchmark_returns: numpy array or pandas Series of returns for the benchmark
    :return: float, the calculated Information Ratio\
    """
    active_returns = returns - (benchmark_returns / 252)
    annualized_active_return = np.mean(active_returns) * 252
    tracking_error = np.std(active_returns) * np.sqrt(252)
    info_ratio = annualized_active_return / tracking_error
    return info_ratio


def calculate_correlation_with_market(portfolio_data, market_data):
    """
    Calculate the correlation between the returns of a portfolio and the market.
    :param DataFrame portfolio_data: DataFrame containing returns of the portfolio
    :param DataFrame market_data: DataFrame containing returns of the market index
    :return: Correlation value
    """
    # Ensuring both dataframes have the same date index
    common_dates = portfolio_data.index.intersection(market_data.index)
    portfolio_data = portfolio_data.loc[common_dates]
    market_data = market_data.loc[common_dates]

    # Calculating correlation
    return portfolio_data.corrwith(market_data)