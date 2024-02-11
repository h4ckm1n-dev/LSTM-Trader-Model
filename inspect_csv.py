import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('./bitcoin_price_data_extended.csv', parse_dates=['Timestamp'], index_col='Timestamp')

# Function to print section headers with blue color
def print_header(header_text, color='blue'):
    color_code = {
        'blue': '\033[34m',  # Setting color to blue
    }
    reset = '\033[0m'
    print(f"{color_code[color]}{'='*60}\n{header_text}\n{'='*60}{reset}")

# Function to print basic information about the dataset
def print_basic_info():
    print_header("Basic Information about the Dataset")

    # Printing basic information about the dataset
    print(df.info())

# Function to check for missing values
def print_missing_values():
    print_header("Missing Values")

    # Printing missing values
    print(df.isnull().sum())

# Function to print summary statistics
def print_summary_statistics():
    print_header("Summary Statistics")

    # Printing summary statistics
    print(df.describe())

# Function to visualize the distribution of features
def visualize_distribution():
    print_header("Distribution of Close Prices")

    # Visualizing the distribution of close prices
    plt.figure(figsize=(12, 8))
    sns.histplot(df['Close'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Close Prices')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    plt.savefig('distribution_of_close_prices.png')

# Call the functions to print and visualize
print_basic_info()
print_missing_values()
print_summary_statistics()
visualize_distribution()
