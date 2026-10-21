import numpy as np

def find_max_non_outlier(x, y, k=2):
    """
    Finds the maximum stock price that isn't an outlier based on the LSRL residuals.

    Parameters:
    - x: NumPy array of time data.
    - y: NumPy array of stock prices.
    - k: Threshold multiplier for standard deviation (default is 2).

    Returns:
    - max_value: Maximum non-outlier stock price.
    - max_time: Corresponding time for the maximum non-outlier stock price.
    """
    # Ensure arrays are NumPy arrays of type float64
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = x.size
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x * x)
    Sxy = np.sum(x * y)

    denominator = n * Sxx - Sx * Sx
    if denominator == 0:
        raise ValueError("Denominator in slope calculation is zero. Check your x data for duplicates.")

    m = (n * Sxy - Sx * Sy) / denominator
    b = (Sy - m * Sx) / n

    predicted_y = m * x + b
    residuals = y - predicted_y

    mean_residual = np.mean(residuals)
    variance = np.mean((residuals - mean_residual) ** 2)
    sigma = np.sqrt(variance)

    upper_threshold = mean_residual + k * sigma
    lower_threshold = mean_residual - k * sigma

    non_outlier_mask = (residuals >= lower_threshold) & (residuals <= upper_threshold)
    non_outlier_y = y[non_outlier_mask]

    if non_outlier_y.size > 0:
        max_value = np.max(non_outlier_y)
        max_index = np.argmax(non_outlier_y)
        max_time = x[non_outlier_mask][max_index]
        return max_value, max_time
    else:
        return None, None

# Example usage:
# x = np.array([...])
# y = np.array([...])
# max_price, max_time = find_max_non_outlier(x, y)
# print(f"Max Non-Outlier Price: {max_price} at time {max_time}")