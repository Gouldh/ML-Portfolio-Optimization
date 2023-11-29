import yfinance as yf
import pandas as pd
from statsmodels.api import OLS, add_constant

# Currently not implemented. Implementation in a future update

def download_stock_data(tickers, start_date, end_date):
    """
    Downloads data for a list of stock ticker strings
    :param str start_date: start date for download in form 'YYYY-MM-DD'
    :param str end_date: end date for download in form 'YYYY-MM-DD'
    :param list tickers: list of stock tickers to gather data for
    :return: pandas dataframe with stock data
    """
    return yf.download(tickers, start_date, end_date, progress=False)


def download_factor_data(start_date, end_date):
    """
    Downloads proxy data for Fama-French three-factor model
    :param string start_date: start date for data in form 'YYYY-MM-DD'
    :param string end_date: end date for data in form 'YYYY-MM-DD'
    :return: pandas DataFrame containing factor data
    """
    # Define ETFs as proxies for the factors
    factor_proxies = {
        'Market': 'SPY',  # Proxy for market risk
        'SMB': ('IJR', 'IVV'),  # Small cap minus large cap
        'HML': ('IVE', 'IVW')   # Value minus growth
    }

    factor_data = pd.DataFrame()

    for factor, tickers in factor_proxies.items():
        if isinstance(tickers, str):
            # Single ticker (e.g., for market)
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close'].pct_change()
            factor_data[factor] = data
        else:
            # Difference between two tickers (e.g., SMB, HML)
            data1 = yf.download(tickers[0], start=start_date, end=end_date, progress=False)['Adj Close'].pct_change()
            data2 = yf.download(tickers[1], start=start_date, end=end_date, progress=False)['Adj Close'].pct_change()
            factor_data[factor] = data1 - data2

    return factor_data.dropna()


def analyze_factor_impact(tickers, start_date, end_date):
    """
    Analyzes the impact of Fama-French factors on the portfolio returns
    :param list tickers: List of stock tickers
    :param string start_date: Start date for analysis in form 'YYYY-MM-DD'
    :param string end_date: End date for analysis in form 'YYYY-MM-DD'
    :return: DataFrame with regression analysis results
    """
    stock_data = download_stock_data(tickers, start_date, end_date)
    factor_data = download_factor_data(start_date, end_date)

    # Calculate daily returns for the stocks
    stock_returns = stock_data['Adj Close'].pct_change().dropna()

    # Merge stock returns with factor data
    merged_data = stock_returns.join(factor_data).dropna()

    results = {}
    for ticker in tickers:
        # Running a linear regression for each stock against the factors
        Y = merged_data[ticker]
        X = merged_data[factor_data.columns]
        X = add_constant(X)

        model = OLS(Y, X).fit()
        results[ticker] = model.summary()

    return results
