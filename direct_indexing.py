import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_market_data(file_path='historicals.csv'):
    """
    Load market data from CSV file and pivot it
    
    Args:
        file_path: Path to the CSV file containing market data with columns (symbol, date, close)
        
    Returns:
        DataFrame with dates as index, symbols as columns, and close prices as values
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Pivot the data
    pivoted_df = df.pivot(index='date', columns='symbol', values='close')
    
    # Sort the index (dates) for time-series consistency
    pivoted_df = pivoted_df.sort_index()
    
    return pivoted_df

def optimize_weights(returns_data, market_returns, max_securities=50):
    """
    Optimize weights for a subset of securities to track the market index
    
    Args:
        returns_data: DataFrame of security returns
        market_returns: Series of market index returns
        max_securities: Maximum number of securities to include in the portfolio
        
    Returns:
        Array of optimized weights
    """
    num_securities = returns_data.shape[1]
    
    # Define the objective function (tracking error)
    def objective(weights):
        portfolio_returns = returns_data.dot(weights)
        tracking_error = np.sum((portfolio_returns - market_returns) ** 2)
        return tracking_error
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
    ]
    
    # Define bounds (each weight between 0 and 1)
    bounds = tuple((0, 1) for _ in range(num_securities))
    
    # Initial equal weights for all securities
    initial_weights = np.ones(num_securities) / num_securities
    
    # Optimize weights
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Get optimized weights
    optimized_weights = result['x']
    
    # If we want to limit the number of securities, keep only the top weights
    if max_securities < num_securities:
        # Get indices of the top weights
        top_indices = np.argsort(optimized_weights)[-max_securities:]
        
        # Create a new weight array with zeros for non-selected securities
        limited_weights = np.zeros(num_securities)
        limited_weights[top_indices] = optimized_weights[top_indices]
        
        # Normalize to sum to 1
        limited_weights = limited_weights / np.sum(limited_weights)
        
        # Re-optimize with the limited set
        bounds = [(0, 1) if i in top_indices else (0, 0) for i in range(num_securities)]
        result = minimize(
            objective,
            limited_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        optimized_weights = result['x']
    
    return optimized_weights

def calculate_portfolio_performance(prices, weights):
    """
    Calculate the performance of a portfolio given prices and weights
    
    Args:
        prices: DataFrame of security prices
        weights: Array of weights for each security
        
    Returns:
        Series of portfolio values
    """
    normalized_prices = prices / prices.iloc[0]
    portfolio_values = normalized_prices.dot(weights)
    return portfolio_values

def calculate_tracking_error(portfolio_values, index_values):
    """
    Calculate tracking error between portfolio and index
    
    Args:
        portfolio_values: Series of portfolio values
        index_values: Series of index values
        
    Returns:
        Tracking error (RMSE)
    """
    # Normalize to same starting point
    norm_portfolio = portfolio_values / portfolio_values.iloc[0]
    norm_index = index_values / index_values.iloc[0]
    
    # Calculate tracking error
    tracking_error = np.sqrt(np.mean((norm_portfolio - norm_index) ** 2))
    return tracking_error

def plot_comparison(dates, portfolio_values, index_values, security_weights):
    """
    Plot the performance of the portfolio vs the index
    
    Args:
        dates: Array of dates
        portfolio_values: Array of portfolio values
        index_values: Array of index values
        security_weights: Dictionary of security names and their weights
    """
    plt.figure(figsize=(12, 8))
    
    # Plot normalized values
    plt.subplot(2, 1, 1)
    norm_portfolio = portfolio_values / portfolio_values[0]
    norm_index = index_values / index_values[0]
    
    plt.plot(dates, norm_portfolio, label='Direct Indexing Portfolio', linewidth=2)
    plt.plot(dates, norm_index, label='Market Index', linewidth=2, linestyle='--')
    plt.title('Direct Indexing Performance vs Market Index')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot weights
    plt.subplot(2, 1, 2)
    securities = list(security_weights.keys())
    weights = list(security_weights.values())
    
    # Only show securities with non-zero weights
    non_zero_indices = [i for i, w in enumerate(weights) if w > 0.001]
    securities = [securities[i] for i in non_zero_indices]
    weights = [weights[i] for i in non_zero_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(securities)))
    plt.bar(securities, weights, color=colors)
    plt.title('Security Weights in Direct Indexing Portfolio')
    plt.xlabel('Security')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('direct_indexing_results.png')
    plt.show()

def main():
    # Load market data
    market_data = load_market_data()
    
    # All columns are securities in the pivoted data
    security_columns = market_data.columns.tolist()
    security_prices = market_data
    
    # Use HXT as the market index instead of calculating an equal-weighted basket
    market_index = security_prices['HXT']
    
    # Remove HXT from the security columns and prices if we don't want to include it in our portfolio
    if 'HXT' in security_columns:
        security_columns.remove('HXT')
        security_prices = security_prices[security_columns]
    
    # Calculate returns (daily percentage changes)
    security_returns = security_prices.pct_change().dropna()
    market_returns = market_index.pct_change().dropna()
    
    # Optimize weights for direct indexing portfolio
    # Note: limiting to 20 securities, can be adjusted as needed
    optimized_weights = optimize_weights(security_returns, market_returns, max_securities=20)
    
    # Create a dictionary of security names and their weights
    security_weight_dict = {security: weight for security, weight in zip(security_columns, optimized_weights)}
    
    # Print the selected securities and their weights
    print("Optimized Direct Indexing Portfolio:")
    for security, weight in sorted(security_weight_dict.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.001:  # Only show non-zero weights
            print(f"{security}: {weight:.4f}")
    
    # Calculate portfolio performance
    portfolio_values = calculate_portfolio_performance(security_prices, optimized_weights)
    
    # Calculate tracking error
    tracking_error = calculate_tracking_error(portfolio_values, market_index)
    print(f"\nTracking Error (RMSE): {tracking_error:.6f}")
    
    # Calculate correlation between portfolio and index
    correlation = np.corrcoef(portfolio_values, market_index)[0, 1]
    print(f"Correlation with Market Index: {correlation:.6f}")
    
    # Plot results
    plot_comparison(security_prices.index, portfolio_values, market_index, security_weight_dict)
    
    # Save the optimized portfolio to CSV for reference
    result_df = market_data.copy()
    result_df['Direct_Indexing_Portfolio'] = portfolio_values
    result_df.to_csv('direct_indexing_results.csv')

if __name__ == "__main__":
    main()
