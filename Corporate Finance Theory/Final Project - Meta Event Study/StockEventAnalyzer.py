import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import t
from pandas.tseries.offsets import BDay
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import scipy.stats as stats
from IPython.display import Markdown, display
from datetime import datetime

# Define a custom color palette
custom_palette = ['#08F7FE', '#FE53BB', '#F5D300', '#00ff41']  # Teal, Pink, Yellow, Neon Green
sns.set_palette(custom_palette)

sns.set_style("darkgrid", {
    "axes.facecolor": "#2c2c2c",
    "grid.color": "#4d4d4d",
    "grid.linestyle": "--",
    "axes.labelcolor": "#fc8d62",
    "axes.edgecolor": "#66c2a5",
    "xtick.color": "#66c2a5",
    "ytick.color": "#66c2a5",
    "axes.titlecolor": "#f4c542",
    "text.color": "#66c2a5"
})


def style_plot(ax, title, ylabel='', xlabel='Date'):
    ax.set_title(title, color="#f4c542", fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, color="#fc8d62", fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel, color="#fc8d62", fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#66c2a5')

    for spine in ax.spines.values():
        spine.set_color("#66c2a5")
    ax.set_facecolor("#2c2c2c")
    ax.grid(True, which='both', color="#4d4d4d", linestyle="--", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=8, facecolor="#2c2c2c", edgecolor="#2c2c2c")

def style_total(ax):
    for spine in ax.spines.values():
        spine.set_color("#66c2a5")
    ax.set_facecolor("#2c2c2c")
    ax.grid(True, which='both', color="#4d4d4d", linestyle="--", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=8, facecolor="#2c2c2c", edgecolor="#2c2c2c")



class StockEventAnalyzer:
    def __init__(self, stock_symbol, market_symbol, bef_event=5, aft_event=5, window_offset=30, window_size=180):
        self.stock_symbol = stock_symbol
        self.market_symbol = market_symbol
        self.bef_event = bef_event
        self.aft_event = aft_event
        self.window_offset = window_offset
        self.window_size = window_size

    def get_trading_days(self, start_date, end_date):
        return pd.date_range(start=start_date, end=end_date, freq=BDay())

    def download_and_calculate_returns(self, start_date, end_date):
        stock_data = yf.download(self.stock_symbol, start=start_date, end=end_date, progress=False)
        market_data = yf.download(self.market_symbol, start=start_date, end=end_date, progress=False)
        
        stock_returns = stock_data['Adj Close'].pct_change()
        market_returns = market_data['Adj Close'].pct_change()
        
        return stock_returns, market_returns
    
    def calculate_statistics_and_significance(self, abnormal_returns, estimation_window_length, event_length, alpha=0.05):
        # Calculate average abnormal return (AAR)
        CAR = abnormal_returns.sum()
        AAR = CAR / len(abnormal_returns)

        # Calculate the t-statistic for individual abnormal returns
        AR_squared = np.square(abnormal_returns)
        variance_AR = 1/(estimation_window_length - 1) * AR_squared.sum()
        std_dev_AR = np.sqrt(variance_AR)
        t_statistic_AR = abnormal_returns / std_dev_AR

        # Calculate the t-statistic for CAR
        variance_CAR = event_length * variance_AR
        std_dev_CAR = np.sqrt(variance_CAR)
        t_statistic_CAR = CAR / std_dev_CAR

        # Calculate the critical t-value for a two-tailed test
        df = estimation_window_length - 1
        t_crit = t.ppf(1 - alpha/2, df)
        #check for significance
        t_statistic_AAR = t_statistic_AR.mean()
        is_significant_AR = np.abs(t_statistic_AAR) > t_crit
        is_significant_CAR = np.abs(t_statistic_CAR) > t_crit

        return AAR, t_statistic_AR, t_statistic_CAR, std_dev_AR, std_dev_CAR, is_significant_AR, is_significant_CAR, t_crit

    def get_event_window(self, event_date):
        event_date = pd.to_datetime(event_date)
        event_window_start = event_date - BDay(self.bef_event) 
        event_window_end = event_date + BDay(self.aft_event)
        est_window_start = event_window_start - BDay(self.window_size)
        est_window_end = event_window_start - BDay(self.window_offset)
        return event_window_start, event_window_end, est_window_start, est_window_end


    def analyze_event(self, event):
        event_name = event['event']
        event_date = pd.to_datetime(event['date'])
        event_window_start, event_window_end, est_window_start, est_window_end = self.get_event_window(event_date)

        stock_returns, market_returns = self.download_and_calculate_returns(est_window_start, event_window_end)

        # Calculation of beta
        beta = self.calculate_beta(stock_returns, market_returns, est_window_start, est_window_end)

        # Isolating the event window stock and market returns
        event_window_stock = stock_returns[(stock_returns.index >= event_window_start) & (stock_returns.index <= event_window_end)]
        event_window_market = market_returns[(market_returns.index >= event_window_start) & (market_returns.index <= event_window_end)]

        # Expected and abnormal returns
        expected_returns = beta * event_window_market
        # is this how to calculate expected returns
        abnormal_returns = event_window_stock - expected_returns

        # Statistics
        stock_length = len(stock_returns[(stock_returns.index >= est_window_start) & (stock_returns.index <= est_window_end)])
        AAR, t_stat_AR, t_stat_CAR, std_dev_AR, std_dev_CAR, is_significant_AR, is_significant_CAR, t_crit = self.calculate_statistics_and_significance(abnormal_returns, stock_length, len(event_window_stock))
        self.display_results(beta, abnormal_returns, event_name, event_date, AAR, t_stat_AR, t_stat_CAR, std_dev_AR, std_dev_CAR, t_crit, is_significant_AR, is_significant_CAR, expected_returns, event_window_stock)

        # Create a 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), facecolor='#2c2c2c')  
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Explicitly set the facecolor for each subplot
        for ax in axs.flatten():
            ax.set_facecolor("#2c2c2c")  
            style_total(ax)  
            
        self.plot_returns(axs[0, 0], event_window_stock, expected_returns, abnormal_returns, event_name, event_date)
        self.plot_abnormal_returns(axs[0, 1], abnormal_returns, event_name, event_date)
        self.plot_cumulative_returns(axs[1, 0], event_window_stock, expected_returns, abnormal_returns, event_name, event_date)
        self.plot_violin_abnormal_returns(axs[1, 1], abnormal_returns, event_date, self.window_size)

        plt.show()

    def calculate_beta(self, stock_returns, market_returns, start, end):
        estimation_window_stock = stock_returns[(stock_returns.index >= start) & (stock_returns.index < end)]
        estimation_window_market = market_returns[(market_returns.index >= start) & (market_returns.index < end)]
        correlation = np.corrcoef(estimation_window_stock.dropna(), estimation_window_market.dropna())[0, 1]
        covar = correlation * np.std(estimation_window_stock) * np.std(estimation_window_market)
        return covar / np.var(estimation_window_market)


    def plot_returns(self, axis, actual_returns, expected_returns, abnormal_returns, event_name, event_date):
        dates_with_data = actual_returns.dropna().index
        axis.set_xticks(dates_with_data)
        axis.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates_with_data], rotation=45, fontsize=8)
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol='%'))

        sns.lineplot(data=actual_returns, label=f'{self.stock_symbol} Actual Returns', ax=axis, color='#08F7FE') 
        sns.lineplot(data=expected_returns, label='Expected Returns', ax=axis, color='#FE53BB')  
        sns.lineplot(data=abnormal_returns, label='Abnormal Returns', linestyle='--', alpha=0.5, ax=axis, color='#F5D300')  
        axis.axvline(x=event_date, color='red', linestyle='--', label='Event Date')
        style_plot(axis, f'{event_name} - Actual vs. Expected Returns around {event_date.strftime("%Y-%m-%d")}', ylabel='Returns (%)')
        axis.set_ylabel('Returns')
        axis.set_xlabel('')
        axis.legend()

    
    def plot_violin_abnormal_returns(self, axis, abnormal_returns, event_date, window_size):
        # Adjust to consider only trading days
        window_start = event_date - BDay(window_size)
        window_end = event_date + BDay(window_size)
        df = abnormal_returns.to_frame(name='Abnormal Returns')
        df['Period'] = 'Before Event'
        df.loc[event_date:, 'Period'] = 'After Event'
        df = df[(df.index >= window_start) & (df.index <= window_end)]

        # Plot the violin plot
        sns.violinplot(x='Period', y='Abnormal Returns', data=df, ax=axis, palette=['#008080', '#FFD700'])#'#572855' 
        style_plot(axis, 'Abnormal Returns Distribution around the Event', ylabel='Abnormal Returns (%)')
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol='%'))
        axis.axhline(y=0, color='red', linestyle='--', linewidth=2)
        axis.set_xlabel('')
    
    def plot_cumulative_returns(self, axis, actual_returns, expected_returns, abnormal_returns, event_name, event_date):
        # Use trading days for event window start
        event_window_start = event_date - BDay(self.bef_event)

        actual_cumulative = (1 + actual_returns[(actual_returns.index >= event_window_start)]).cumprod()
        expected_cumulative = (1 + expected_returns[(expected_returns.index >= event_window_start)]).cumprod()
        abnormal_cumulative = (1 + abnormal_returns[(abnormal_returns.index >= event_window_start)]).cumprod()

        # Normalize both series to start at 1
        actual_cumulative /= actual_cumulative.iloc[0]
        expected_cumulative /= expected_cumulative.iloc[0]
        abnormal_cumulative /= abnormal_cumulative.iloc[0]

        # Only show x-ticks for dates with data
        dates_with_data = actual_cumulative.dropna().index
        axis.set_xticks(dates_with_data)
        axis.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates_with_data], rotation=45, fontsize=8)


        sns.lineplot(data=actual_cumulative, label='Actual Cumulative Returns', ax=axis, color='#08F7FE')
        sns.lineplot(data=expected_cumulative, label='Expected Cumulative Returns', ax=axis, color='#FE53BB')
        sns.lineplot(data=abnormal_cumulative, label='Cumulative Abnormal Returns', linestyle='--', alpha=0.5, ax=axis, color='#F5D300')
        axis.axvline(x=event_date, color='red', linestyle='--', label='Event Date')
        style_plot(axis, f'{event_name} - Cumulative Returns vs. Expected around {event_date.strftime("%Y-%m-%d")}', ylabel='Normalized Cumulative Returns')
        axis.set_xlabel('')
        axis.legend()

    def plot_abnormal_returns(self, axis, abnormal_returns, event_name, event_date):
        # Use trading days for event window
        event_window_start = event_date - BDay(self.bef_event)
        event_window_end = event_date + BDay(self.aft_event)
        abnormal_returns = abnormal_returns[(abnormal_returns.index >= event_window_start) & (abnormal_returns.index <= event_window_end)]
        sns.barplot(x=abnormal_returns.index.strftime('%Y-%m-%d'), y=abnormal_returns.values, ax=axis, palette=['#FE53BB' if val < 0 else '#F5D300' for val in abnormal_returns.values])
        style_plot(axis, f'{event_name} - Abnormal Returns around {event_date.strftime("%Y-%m-%d")}', ylabel='Abnormal Returns (%)')
        axis.set_xlabel('')
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol='%'))
        axis.set_xticklabels(abnormal_returns.index.strftime('%Y-%m-%d'), rotation=45, fontsize=8)
        axis.set_ylabel('Abnormal Returns')

    def display_results(self, beta, abnormal_returns, event_name, event_date, AAR, t_stat_AR, t_stat_CAR, std_dev_AR, std_dev_CAR, t_crit, is_significant_AR, is_significant_CAR, expected_returns, event_window_stock):
        CAR = abnormal_returns.sum()
        df = len(abnormal_returns) - 1

        # Formatting returns as a Markdown table
        returns_md = "| Date | Expected Return | Actual Return | Abnormal Return |\n| --- | --- | --- | --- |\n"
        for date, exp_ret, act_ret, ab_ret in zip(expected_returns.index, expected_returns, event_window_stock, abnormal_returns):
            returns_md += f"| {date.strftime('%Y-%m-%d')} | {exp_ret:.2%} | {act_ret:.2%} | {ab_ret:.2%} |\n"

        # Formatting the statistical results as Markdown
        stats_md = f"""
### Event Analysis

| Metric | Value |
| --- | --- |
| **Event** | {event_name} |
| **Base Stock** | {self.stock_symbol} |
| **Market** | {self.market_symbol} |
| **Date** | {event_date.strftime('%Y-%m-%d')} |
| **Beta** | {beta:.2f} |
| **Estimation Window** | {int(self.window_size/2)} days before/after event |
| **Event Window** | {self.aft_event} days before/after event |
| **Observations** | {len(abnormal_returns)} trading days |

### Returns
| Metric | Value |
| --- | --- |
| **Cumulative AR** | {CAR:.2%} |
| **Average AR** | {AAR:.2%} |

### Hypothesis Testing
| Metric | Value |
| --- | --- |
| **Significance Level** | 0.05 |
| **Degrees of Freedom** | {df} |
| **Critical T-value** | {t_crit:.3f} |
| **T-stat for AAR** | {t_stat_AR.mean():.3f} |
| **T-stat for CAR** | {t_stat_CAR:.3f} |
| **SD of AAR, CAR** | {std_dev_AR:.4f}, {std_dev_CAR:.4f} |

### Significance
| Metric | Value |
| --- | --- |
| **Is T-stat for AR significant?** | {'Yes' if is_significant_AR.mean() else 'No'} |
| **Is T-stat for CAR significant?** | {'Yes' if is_significant_CAR else 'No'} |
| **Is CAR positive?** | {'Yes' if CAR > 0 else 'No'} |
| **Is AAR positive?** | {'Yes' if AAR > 0 else 'No'} |

### Expected, Actual, and Abnormal Returns
"""

    # Combine all parts into a single Markdown string
        full_markdown = stats_md + "\n" + returns_md

        # Display using the Markdown class
        display(Markdown(full_markdown))