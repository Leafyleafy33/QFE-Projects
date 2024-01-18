import pandas as pd
import pandas_datareader as pdr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import datetime
import random
import riskfolio as rp
import matplotlib.patches as patches

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform

sns.set_style("darkgrid", {
    "axes.facecolor": "#0a1029",  # deep moonlit night shade
    "grid.color": "#122a50",     # slightly brighter grid lines
    "grid.linestyle": "--",
    "axes.labelcolor": "#eccfac",
    "axes.edgecolor": "#eccfac",
    "xtick.color": "#eccfac",
    "ytick.color": "#eccfac",
    "axes.titlecolor": "#eccfac",
    "text.color": "#eccfac"
})

TICKER_NAME_MAPPING = {
    'SPY': 'S&P 500',
    'TLT': '20+ Year Treasury',
    'UUP': 'US Dollar Index',
    'DBC': 'Commodity Index',
    'VWO': 'Emerging Markets',
    'GLD': 'Gold',
    'DGL': 'DB Gold Fund',
    'IAU': 'Gold Trust',
    'DBP': 'DB Precious Metals Fund',
    'PALL': 'Physical Palladium Shares',
    'RING': 'Global Gold Miners',
    'GDX': 'Gold Miners',
    'ITM': 'AMT-Free Intermediate Municipal',
    'MBB': 'MBS ETF',
    'AGG': 'Core U.S. Aggregate Bond',
    'LQD': 'iBoxx $ Investment Grade Corporate Bond',
    'HYG': 'iBoxx $ High Yield Corporate Bond',
    'EMB': 'JP Morgan USD Emerging Markets Bond',
    'JNK': 'SPDR Barclays High Yield Bond',
    'KWEB': 'CSI China Internet',
    'DBS': 'DB Silver Fund',
    'SIVR': 'Physical Silver Shares',
    'SLV': 'Silver Trust',
    'XLK': 'Technology Select Sector',
    'PPA': 'Aerospace & Defense',
    'IHF': 'U.S. Healthcare Providers',
    'SYLD': 'Multi-Asset Income',
    'XLY': 'Consumer Discretionary Select Sector Fund',
    'XLP': 'Consumer Staples Select Sector Fund',
    'XLE': 'Energy Select Sector Fund',
    'XLF': 'Financial Select Sector Fund',
    'XLI': 'Industrial Select Sector Fund',
    'XLK': 'Technology Select Sector Fund',
    'XLB': 'Materials Select Sector Fund',
    'XME': 'S&P Metals & Mining ETF',
    'XBI': 'SPDR S&P Biotech ETF',
    'XLU': 'Utilities Select Sector Fund',
    'DBB': 'DB Base Metals Fund',
    'QVAL': 'QuantShares U.S. Market Neutral Value',
    'QWLD': 'SPDR MSCI World StrategicFactors',
    'QEMM': 'SPDR MSCI Emerging Markets StrategicFactors',
    'PDBC': 'Optimum Yield Diversified Commodity Strategy',
    'PDP': 'DWA Momentum',
    'PBE': 'Dynamic Biotechnology & Genome',
    'PBJ': 'Dynamic Food & Beverage',
    'PEJ': 'Dynamic Leisure & Entertainment',
    'PJP': 'Dynamic Pharmaceuticals',
    'PSJ': 'Dynamic Software',
}
tickers = list(TICKER_NAME_MAPPING.keys())

class PortfolioAnalysis:
    def __init__(self, path, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.ETF = pd.read_csv(path, parse_dates=['date']).query(f"ticker.isin({tickers})")
        self.Dwide = None
        self.Dwide_FullTime = None
        self.Smat = None
        self.optimized_weights = None
        self.RPP = None
        self.Port = None
        self.PortL = None
        self.LeverageRatio = None

    @staticmethod # 1e-10
    def _minFn(x, C, b):
        return 0.5 * (x.T @ C @ x) - b.T @ np.log(x + 1e-10)
    
    def _get_risk_free_rate(self):
        start = datetime.datetime(self.start_date, 1, 1)
        end = datetime.datetime(self.end_date, 12, 31)

        treasury_yield = pdr.get_data_fred('GS10', start, end)

        risk_free_rate = treasury_yield['GS10'].iloc[-1] / 12 / 100
        return risk_free_rate
    
    def set_plot_style(self):
        sns.set_style("darkgrid", {
            "axes.facecolor": "#0a1029",  # deep moonlit night shade
            "grid.color": "#122a50",     # slightly brighter grid lines
            "grid.linestyle": "--",
            "axes.labelcolor": "#eccfac",
            "axes.edgecolor": "#eccfac",
            "xtick.color": "#eccfac",
            "ytick.color": "#eccfac",
            "axes.titlecolor": "#eccfac",
            "text.color": "#eccfac"
        })
    
    def preprocess(self):
        self.Dwide = self.ETF[(self.ETF['date'].dt.year >= self.start_date) & (self.ETF['date'].dt.year <= self.end_date)].pivot_table(columns='ticker', index='date', values='ret')
        self.Dwide_FullTime = self.ETF[self.ETF['date'].dt.year >= self.start_date].pivot_table(columns='ticker', index='date', values='ret')
        self.Smat = self.Dwide.cov()
        N_assets = len(self.tickers)
        initial_x = np.full(N_assets, 0.05)
        b = np.full(N_assets, 1/N_assets)
        bounds0 = [(0, None) for _ in range(N_assets)]
        myOpt = minimize(self._minFn, x0=initial_x, args=(self.Smat, b), bounds=bounds0)
        W = myOpt['x'] / myOpt['x'].sum()
        RC = W * (self.Smat @ W) / (np.sqrt(W.T @ self.Smat @ W))
        self.RPP = pd.DataFrame({'W': W, 'RC': RC}).reset_index()
        self.market_return = self.Dwide['SPY'].mean()
        self.risk_free_rate = self._get_risk_free_rate()
        self.market_risk_premium = self.market_return - self.risk_free_rate


    def compute_returns(self, weights=None):
        D_OOS = pd.merge(self.ETF.query(f'date.dt.year >= {self.end_date+1}'), self.RPP, on='ticker')
        self.Port = (D_OOS.assign(w_ret=lambda df: df['W'] * df['ret'])
                            .groupby('date')
                            .agg(rpp_ret=('w_ret', 'sum'), ewret=('ret', 'mean'), spret=('sprtrn', 'first'))
                            .reset_index())
        self.LeverageRatio = self.Port['spret'].std() / self.Port['rpp_ret'].std()
        self.Port = self.Port.assign(lev_rpp=lambda df: self.LeverageRatio * df['rpp_ret'])

        # Compute the HERC portfolio's returns using the hierarchical clustering approach
        if weights is None:
            distance_matrix, matrix_d_bar = self.compute_distance_matrices()
            linkage_matrix = self.hierarchical_clustering(matrix_d_bar)
            herc_weights = self.herc_allocation(linkage_matrix, distance_matrix, matrix_d_bar)
        else:
            herc_weights = weights
        D_OOS_HERC = pd.merge(self.ETF[self.ETF['date'].dt.year >= self.start_date], 
                         pd.DataFrame({'ticker': self.Dwide.columns, 'W': herc_weights}), 
                         on='ticker')
        herc_port = D_OOS_HERC.assign(w_ret=lambda df: df['W'] * df['ret']).groupby('date').agg(herc_ret=('w_ret', 'sum')).reset_index()
        self.Port = pd.merge(self.Port, herc_port, on='date')
        self.HercLeverageRatio = self.Port['spret'].std() / self.Port['herc_ret'].std()
        self.Port = self.Port.assign(lev_herc=lambda df: self.HercLeverageRatio * df['herc_ret'])
        self.PortL = (self.Port.melt(id_vars='date', value_name='return', var_name='portfolio')
                                .sort_values(by=['portfolio', 'date'])
                                .assign(ret_plus1=lambda df: 1 + df['return'] / 100, cumRet=lambda df: df.groupby('portfolio')['ret_plus1'].cumprod()))

    def compute_metrics(self):
        beta = self.Dwide.cov().loc[self.tickers, 'SPY'] / self.Dwide['SPY'].var()
        expected_return = (self.risk_free_rate + beta * self.market_risk_premium) / 100
        metrics = {"Beta": beta, "Expected Return": expected_return}
        return metrics

    def plot_cumulative_tickers(self):
        # Compute cumulative returns for each ticker using Dwide_FullTime
        for ticker in self.tickers:
            self.Dwide_FullTime[f'cumulative_return_{ticker}'] = (1 + self.Dwide_FullTime[ticker] / 100).cumprod() - 1

        # Create a dictionary to store the ending values for each ticker
        ending_values = {ticker: self.Dwide_FullTime[f'cumulative_return_{ticker}'].iloc[-1] for ticker in self.tickers}
        sorted_tickers = [k for k, v in sorted(ending_values.items(), key=lambda item: item[1], reverse=True)]
        colors = plt.cm.viridis(np.linspace(1, 0.1, len(sorted_tickers)))

        # Calculate scaling factors based on the number of tickers
        num_tickers = len(sorted_tickers)
        line_width_scaling = 0.7 / num_tickers  # Slightly smaller line width
        font_size_scaling = 8 / num_tickers  # Slightly larger legend font size

        fig, ax = plt.subplots(figsize=(9.7, (10 * 0.618)))
        fig.patch.set_facecolor('#0a1029')

        for i, (ticker, color) in enumerate(zip(sorted_tickers, colors)):
            cumulative_returns = self.Dwide_FullTime[f'cumulative_return_{ticker}']
            label = TICKER_NAME_MAPPING[ticker]
            if label == 'S&P 500':
                cumulative_returns.plot(ax=ax, label=label, linewidth=2, alpha=1, color='red', zorder=3)
            else:
                # Adjust line width and legend font size
                scaled_line_width = 0.7 - i * line_width_scaling
                scaled_legend_font_size = 11.5 - i * font_size_scaling

                cumulative_returns.plot(ax=ax, label=label, linewidth=scaled_line_width, alpha=1, color=color, zorder=2)

        # Adjust the legend font size
        ax.legend(fontsize=scaled_legend_font_size)
        ax.set_title('Cumulative Return of $1 Invested in Each Ticker', color="#eccfac", fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', color="#eccfac", fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return', color="#eccfac", fontsize=14, fontweight='bold')
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.0f}".format(x)))

        # Keep the axis tick label font size fixed
        ax.tick_params(axis='both', which='major', labelsize=10)

        for spine in ax.spines.values():
            spine.set_color("#eccfac")

        plt.show()


    def plot_cumulative_portfolios(self):
        portfolios = {
            'ewret': 'Equal-Weight',
            'rpp_ret': 'Risk Parity',
            'lev_rpp': 'Leveraged Parity',
            'spret': 'S&P 500',
            'herc_ret': 'HERC',
            'lev_herc': 'Leveraged HERC'
        }

        fig, ax = plt.subplots(figsize=(9.7, (10* 0.618)))
        fig.patch.set_facecolor('#0a1029')
        sns.lineplot(data=self.PortL, x='date', y='cumRet', hue='portfolio')
        plt.axhline(y=1, linestyle='--', color='red')
        plt.title('Return to $1 Investment in Portfolios')
        ax.legend([portfolios[key] for key in portfolios.keys()], fontsize=6)
        ax.set_title(f'Cumulative Return of $1 Invested for portfolio {self.end_date + 1}', color="#eccfac", fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', color="#eccfac", fontsize=14, fontweight='bold')  # Added 'Date' as xlabel
        ax.set_ylabel('Cumulative Return', color="#eccfac", fontsize=14, fontweight='bold')  # Added ylabel for clarity
        ax.tick_params(axis='both', which='major', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#eccfac")

        plt.show()

    def statistics(self):
        #rfr = self._get_risk_free_rate() # Get the risk-free rate for when that's relevant
        stats_df = (self.PortL.groupby('portfolio')
                               .agg(Avg=('return', 'mean'), StdDev=('return', 'std'),
                                    # Sharpe with a risk-free rate:
                                    # Sharpe=('return', lambda s: (s.mean() - rfr) / s.std()),
                                    # Version from Class notes:
                                    Sharpe=('return', lambda s: s.mean() / s.std()),
                                    VaR1=('return', lambda s: s.quantile(0.01)),
                                    VaR5=('return', lambda s: s.quantile(0.05)), cumret=('cumRet', lambda s: s.iloc[-1])))
        for portfolio in stats_df.index:
            downside_std = self.PortL.loc[(self.PortL['portfolio'] == portfolio) & (self.PortL['return'] < 0), 'return'].std()
            # Sortino with a risk-free rate:
            # stats_df.loc[portfolio, 'Sortino'] = (stats_df.loc[portfolio, 'Avg'] - rfr) / downside_std
            # Version from Class notes:
            stats_df.loc[portfolio, 'Sortino'] = stats_df.loc[portfolio, 'Avg'] / downside_std
        return stats_df

    def plot_sml(self, tickers):
        all_betas = [self.compute_metrics()["Beta"][ticker] for ticker in tickers]
        min_beta, max_beta = min(all_betas), max(all_betas)
        abs_max_beta = max(abs(min(all_betas)), abs(max(all_betas)))
        beta_values = np.linspace(-abs_max_beta - 0.2, abs_max_beta + 0.2, 200)
        sml = (self.risk_free_rate + beta_values * self.market_risk_premium) / 100
        fig, ax = plt.subplots(figsize=(9.7, (10* 0.618)))
        fig.patch.set_facecolor('#0a1029')

        ax.plot(beta_values, sml, '-', color="#65087e", linewidth=2.5, zorder=2)
        occupied_positions = set()

        def is_occupied(x, y, buffer=0.003):
            for dx in np.arange(-buffer, buffer, buffer):
                for dy in np.arange(-buffer, buffer, buffer):
                    if (x + dx, y + dy) in occupied_positions:
                        return True
            return False

        def find_unoccupied_position(x, y, step_size=0.001):
            theta = 0
            while True:
                dx = step_size * np.cos(theta)
                dy = step_size * np.sin(theta)
                if not is_occupied(x + dx, y + dy):
                    return x + dx, y + dy
                theta += np.radians(40)
                if theta > np.radians(360):
                    theta = 0
                    step_size += 0.001

        num_tickers = len(tickers)
        label_scaling_factor = 6 / (num_tickers*(1/2))  # Adjust label size scaling
        dot_size_scaling_factor = 350 / (num_tickers*(1/6))  # Adjust dot size scaling

        for ticker in tickers:
            dx_variance = random.uniform(-0.05, 0.02) * (max_beta - min_beta)
            dy_variance = random.uniform(-0.05, 0.1) * (max(sml) - min(sml))
            dx = (max_beta - min_beta) * 0.04 + dx_variance
            dy = (max(sml) - min(sml)) * 0.04 + dy_variance

            beta = self.compute_metrics()["Beta"][ticker]
            exp_return = self.compute_metrics()["Expected Return"][ticker]
            color_intensity = (exp_return - min(sml)) / (max(sml) - min(sml))
            point_color = plt.cm.coolwarm_r(color_intensity)

            if np.random.choice([True, False]):
                quadrant = 'UL' if np.random.rand() < 0.5 else 'LR'
            else:
                quadrant = 'LR' if np.random.rand() < 0.5 else 'UL'

            if quadrant == 'UL':
                dx_val, dy_val = find_unoccupied_position(-dx, dy + 0.0085)
            else:
                dx_val, dy_val = find_unoccupied_position(dx, -dy - 0.0085)

            occupied_positions.add((beta + dx_val, exp_return + dy_val))

            # Adjust dot size and label font size
            scaled_dot_size = dot_size_scaling_factor if ticker == 'SPY' else dot_size_scaling_factor / 2
            scaled_label_size = 10 if ticker == 'SPY' else 6 - label_scaling_factor

            ax.scatter(beta, exp_return, s=scaled_dot_size, color=point_color, edgecolor="#d7357b" if ticker != 'SPY' else "#f5da42", linewidth=1.5, zorder=5)
            ax.text(beta + dx_val, exp_return + dy_val, TICKER_NAME_MAPPING[ticker], fontsize=scaled_label_size, verticalalignment='center', horizontalalignment='center', color=point_color, fontweight='bold')
            ax.plot([beta, beta + dx_val], [exp_return, exp_return + dy_val], color=point_color, linewidth=0.5, linestyle="--")

        ax.set_title('Security Market Line (SML)', color="#eccfac", fontsize=16, fontweight='bold')
        ax.set_xlabel('Beta', color="#eccfac", fontsize=8, fontweight='bold')
        ax.set_ylabel('Expected Return', color="#eccfac", fontsize=8, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color("#eccfac")

        plt.show()
    

    def plot_correlation_matrix(self):
        #correlation_matrix = self.Dwide.corr()
        corr_matrix = self.Dwide.corr()
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Convert the 2D square matrix to a 1D condensed matrix
        condensed_dist_matrix = squareform(distance_matrix)

        # Compute the linkage matrix using the condensed distance matrix with the Ward method
        linkage_mat = linkage(condensed_dist_matrix, method='complete')

        # Get the order of assets based on hierarchical clustering
        order = leaves_list(linkage_mat)

        # Reorder the covariance matrix
        quasi_diag_cov_matrix = self.Smat.iloc[order, :].copy()
        quasi_diag_cov_matrix = quasi_diag_cov_matrix.iloc[:, order]
        # Dynamically adjust figure size based on number of tickers
        correlation_matrix = quasi_diag_cov_matrix.corr()
        fig_scaling_factor = len(correlation_matrix.columns) * 0.35
        plt.figure(figsize=(fig_scaling_factor + 3, fig_scaling_factor), facecolor='#0a1029')
        plt.gca().add_patch(plt.Rectangle((0, 0), len(correlation_matrix.columns), len(correlation_matrix.columns), fc="none", ec="#eccfac", lw=1))

        # Inverting the color palette for the correlation matrix
        cmap = sns.diverging_palette(12, 250, s=75, l=40, n=9, center="dark", as_cmap=True)

        # Heatmap with the updated color palette
        cbar = sns.heatmap(correlation_matrix, annot=True, cmap=cmap, linewidths=1, linecolor='#0a1029', 
                        cbar_kws={"shrink": 0.45, "ticks": [-1, -0.5, 0, 0.5, 1]}, vmin=-1, vmax=1, square=True, fmt=".1f", annot_kws={"color": "#eccfac", "weight": "bold", "fontsize": 6.5}, xticklabels=True, yticklabels=True).collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)

        plt.title('Correlation Matrix', fontsize=18, fontweight='bold', color='#eccfac')
        plt.xlabel('', fontsize=12, fontweight='bold', color='#eccfac')
        plt.ylabel('', fontsize=12, fontweight='bold', color='#eccfac')
        plt.xticks(color="#eccfac", fontsize=8, rotation=45)
        plt.yticks(color="#eccfac", fontsize=8)

        plt.show()
            
    def report(self):
        SEPARATOR = "-" * 120

        # Header
        print("Portfolio Analysis Report".center(120))
        print(SEPARATOR)
        print()

        # General Info
        print("Tickers: ", end="")
        tickers_str = ', '.join(sorted(self.tickers))  # sort tickers alphabetically
        max_ticker_length = len(SEPARATOR) - len("Tickers: ")

        while len(tickers_str) > max_ticker_length:
            # find the last comma before the limit
            idx = tickers_str.rfind(',', 0, max_ticker_length)
            print(tickers_str[:idx], end=",")
            tickers_str = tickers_str[idx+1:].strip()
            if len(tickers_str) > 0:
                print("\n" + " " * 9, end="")

        print(tickers_str)
        print(f"Date: {self.start_date} to {self.end_date}")
        print(SEPARATOR)
        print()

        # Optimized Portfolio Weights, Risk Contributions, and Metrics
        print("Weights, Risk Contributions & Metrics:")
        headers = ["Ticker", "Weight", "Risk Contrib.", "Beta", "Return"]
        print("{:<15} {:<18} {:<18} {:<18} {:<18}".format(*headers))
        print(SEPARATOR)

        metrics = self.compute_metrics()
        for ticker, weight, rc in zip(self.RPP['ticker'], self.RPP['W'], self.RPP['RC']):
            beta = metrics['Beta'][ticker]
            ret = metrics['Expected Return'][ticker]
            print(f"{ticker:<15} {weight:<18.4f} {rc:<18.4f} {beta:<18.4f} {ret:<18.4f}")
        print(SEPARATOR)
        print()

        # Portfolio Statistics
        print("Portfolio Statistics:")
        stats_df = self.statistics().round(3)
        portfolios = {
            'ewret': 'Equal-Weight',
            'rpp_ret': 'Risk Parity',
            'lev_rpp': 'Leveraged Parity',
            'spret': 'S&P 500'
        }

        # Display the statistics for each portfolio in columns
        headers = ["Metric"] + list(portfolios.values())
        print("{:<15} {:<18} {:<18} {:<18} {:<18}".format(*headers))
        print(SEPARATOR)

        metrics = ['Avg', 'StdDev', 'Sharpe', 'VaR1', 'VaR5', 'Sortino', 'cumret']
        for metric in metrics:
            row_data = [metric]
            for old_name in portfolios.keys():
                row_data.append(stats_df.loc[old_name, metric])
            print("{:<15} {:<18.3f} {:<18.3f} {:<18.3f} {:<18.3f}".format(*row_data))
        print(SEPARATOR)
        print()

class HERCPortfolioAnalysis(PortfolioAnalysis):
    def compute_distance_matrices(self):
        """
        Compute the distance matrices D and D_bar based on historical asset returns.
        """
        corr_matrix = self.Dwide.corr()

        # Convert the correlation matrix to a distance matrix D
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Calculate the D_bar matrix
        matrix_d_bar = squareform(pdist(distance_matrix.values, metric='euclidean'))
        
        return distance_matrix, matrix_d_bar
    
    def hierarchical_clustering(self, matrix_d_bar):
        """
        Perform hierarchical clustering and return the linkage matrix.
        """
        # Convert the 2D square matrix to a 1D condensed matrix
        condensed_matrix_d_bar = squareform(matrix_d_bar)

        # Compute the linkage matrix using the condensed D_bar matrix with single linkage
        linkage_matrix = linkage(condensed_matrix_d_bar, method='complete')
        
        return linkage_matrix

        
    def plot_dendrogram(self, linkage_matrix):
        """Plot the hierarchical clustering dendrogram with a custom legend."""
        
        # Set plot style
        self.set_plot_style()

        # Create a figure with the specified size and background color
        fig, ax = plt.subplots(figsize=(10, (9.5*.618)))
        fig.patch.set_facecolor('#0a1029')
        ax.set_facecolor('#0a1029')
        
        # Determine an appropriate color threshold
        color_thresh = np.average(linkage_matrix[:, 2])
        
        # Plot dendrogram with specified properties
        dend = dendrogram(
            linkage_matrix, 
            ax=ax, 
            labels=self.Dwide.columns, 
            color_threshold=color_thresh, 
            orientation='top', 
            leaf_font_size=7,
            above_threshold_color='#8B8589',
            leaf_label_func=lambda v: str(self.Dwide.columns[v])
        )
        
        # Define the custom legend
        legend_labels = {
            'Emerging Mkt Cluster': '#ff7f0e',  # Orange
            'Equities Cluster': '#2ca02c',      # Green
            'Bond Cluster': '#d62728',          # Red
            'Gold Cluster': '#8c564b',          # Brown
            'Precious Metals Cluster': '#9467bd' # Purple
        }
        
        # Create the custom legend
        #custom_lines = [Line2D([0], [0], color=color, lw=4) for color in legend_labels.values()]
        #ax.legend(custom_lines, legend_labels.keys(), loc='upper right')
        
        # Customize the axes, title, and labels
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold', color="#eccfac")
        ax.set_xlabel('Euclidean Distance', fontsize=14, fontweight='bold', color="#eccfac")
        ax.set_ylabel('Assets', fontsize=14, fontweight='bold', color="#eccfac")
        
        # Configure the appearance of plot spines
        for spine in ax.spines.values():
            spine.set_color("#eccfac")
        
        # Display the plot
        plt.show()
        
    def plot_herc_cumulative_returns(self, weights = None):
        self.set_plot_style()
        
        if weights is None:
            distance_matrix, matrix_d_bar = self.compute_distance_matrices()
            linkage_matrix = self.hierarchical_clustering(matrix_d_bar)
            weights = self.herc_allocation(linkage_matrix, distance_matrix, matrix_d_bar)
        else:
            weights = weights

        D_OOS = pd.merge(self.ETF[self.ETF['date'].dt.year >= self.start_date], 
                         pd.DataFrame({'ticker': self.Dwide.columns, 'W': weights}), 
                         on='ticker')
        herc_port = D_OOS.assign(w_ret=lambda df: df['W'] * df['ret']).groupby('date').agg(herc_ret=('w_ret', 'sum')).reset_index()

    
        # Merge the HERC returns with the other portfolio returns
        combined_port = pd.merge(self.Port, herc_port, on='date')
        
        portfolios = {
            'ewret': 'Equal-Weight',
            'rpp_ret': 'Risk Parity',
            'lev_rpp': 'Leveraged Parity',
            'spret': 'S&P 500',
            'herc_ret': 'HERC',
            'lev_herc': 'Leveraged HERC'
        }
    
        color1 = '#007acc'  
        color2 = '#ff8c00' 
    
        # Create a color gradient
        num_portfolios = len(portfolios)
        colors = [plt.cm.viridis(x) for x in np.linspace(0, 1, num_portfolios)]
    
        # Calculate cumulative returns
        combined_port = combined_port.melt(id_vars='date', value_name='return', var_name='portfolio').sort_values(by=['portfolio', 'date']).assign(ret_plus1=lambda df: 1 + df['return'] / 100, cumRet=lambda df: df.groupby('portfolio')['ret_plus1'].cumprod())
    
        # Plot cumulative returns
        fig, ax = plt.subplots(figsize=(9.7, (10 * 0.618)))
        fig.patch.set_facecolor('#0a1029')
        
        # Plot individual portfolio lines with gradient colors
        for i, (portfolio, _) in enumerate(portfolios.items()):
            sns.lineplot(data=combined_port[combined_port['portfolio'] == portfolio], x='date', y='cumRet', label=portfolios[portfolio], color=colors[i])
        
        plt.axhline(y=1, linestyle='--', color='red')
        plt.title('Return to $1 Investment in Portfolios')
        
        # Create a custom legend
        custom_legend = [plt.Line2D([0], [0], color=colors[i], lw=2, label=portfolios[portfolio]) for i, (portfolio, _) in enumerate(portfolios.items())]
        ax.legend(handles=custom_legend, fontsize=7)
        
        ax.set_title(f'Cumulative Return of $1 Invested for portfolio {self.end_date + 1}', color="#eccfac", fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', color="#eccfac", fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return', color="#eccfac", fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#eccfac")

    def herc_allocation(self, linkage_matrix, distance_matrix, matrix_d_bar):
        """
        Perform the HERC portfolio allocation.
        """
        num_assets = len(distance_matrix)
        weights = np.zeros(num_assets)
        
        # Initially, consider all assets as one cluster
        cluster = [list(range(num_assets))]
        
        while len(cluster) < num_assets:
            # Bisection
            # Find the cluster to split, which is the last cluster in the list
            bisection = fcluster(linkage_matrix, len(cluster) + 1, criterion='maxclust')
            unique_clusters = np.unique(bisection)
            # Find which cluster to split
            cluster_to_split = np.argmax([np.sum(bisection == i) for i in unique_clusters])
            
            # Update the list of clusters
            assets_in_cluster = [i for i in range(num_assets) if bisection[i] == cluster_to_split + 1]
            
            # Correctly remove the cluster to split and add the new subclusters
            cluster = [c for c in cluster if set(c) != set(assets_in_cluster)]
            half = len(assets_in_cluster) // 2
            cluster.extend([assets_in_cluster[:half], assets_in_cluster[half:]])

            # Risk Parity Allocation within clusters
            for subcluster in cluster:
                subcov = self.Smat.iloc[subcluster, subcluster]
                inv_diag = 1 / np.diag(subcov)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                weights[subcluster] = parity_w
        
        return weights / np.sum(weights)  # Normalize the weights to sum to 1
        
    def herc_report(self, weights = None):
        SEPARATOR = "-" * 120
        
        # HERC Portfolio Analysis Report
        print("HERC Portfolio Analysis Report".center(120, '-'))
        print(f"Total Assets in Portfolio: {len(self.tickers)}")
        print(f"Date Range: {self.start_date} to {self.end_date}\n")
    
        # Compute distance matrix and linkage matrix for HERC
        if weights is None:
            distance_matrix, matrix_d_bar = self.compute_distance_matrices()
            linkage_matrix = self.hierarchical_clustering(matrix_d_bar)

            # Plot dendrogram
            self.plot_dendrogram(linkage_matrix)

            # Compute HERC weights
            herc_weights = self.herc_allocation(linkage_matrix, distance_matrix, matrix_d_bar)
        else:
            herc_weights = weights
    
        print(SEPARATOR)
        print()
        
        # Portfolio Analysis Report (from report function)
        print("Portfolio Analysis Report".center(120))
        print(SEPARATOR)
        print()

        # General Info
        print("Tickers: ", end="")
        tickers_str = ', '.join(sorted(self.tickers))
        max_ticker_length = len(SEPARATOR) - len("Tickers: ")
        while len(tickers_str) > max_ticker_length:
            idx = tickers_str.rfind(',', 0, max_ticker_length)
            print(tickers_str[:idx], end=",")
            tickers_str = tickers_str[idx+1:].strip()
            if len(tickers_str) > 0:
                print("\n" + " " * 9, end="")
        print(tickers_str)
        print(f"Date: {self.start_date} to {self.end_date}")
        print(SEPARATOR)
        print()

        # Optimized Portfolio Weights, Risk Contributions, and Metrics with HERC Weights
        print("Weights, Risk Contributions, HERC RC & Metrics:")
        headers = ["Ticker", "Weight", "HERC Weights", "Risk Contrib.", "Beta", "Return"]
        print("{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(*headers))
        print(SEPARATOR)

        metrics = self.compute_metrics()
        for ticker, weight, rc, h_weight in zip(self.RPP['ticker'], self.RPP['W'], self.RPP['RC'], herc_weights):
            beta = metrics['Beta'][ticker]
            ret = metrics['Expected Return'][ticker]
            print(f"{ticker:<15} {weight:<15.4f} {h_weight:<15.4f} {rc:<15.4f} {beta:<15.4f} {ret:<15.4f}")
        print(SEPARATOR)
        print()

        # Portfolio Statistics with HERC Portfolio
        print("Portfolio Statistics:")
        stats_df = self.statistics().round(3)
        portfolios = {
            'ewret': 'Equal-Weight',
            'rpp_ret': 'Risk Parity',
            'lev_rpp': 'Leveraged Parity',
            'spret': 'S&P 500',
            'herc_ret': 'HERC',
            'lev_herc': 'Leveraged HERC'
        }

        headers = ["Metric"] + list(portfolios.values())
        print("{:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(*headers))

        print(SEPARATOR)

        metrics = ['Avg', 'StdDev', 'Sharpe', 'VaR1', 'VaR5', 'Sortino', 'cumret']
        for metric in metrics:
            row_data = [metric]
            for old_name in portfolios.keys():
                row_data.append(stats_df.loc[old_name, metric])
            print("{:<15} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format(*row_data))
        print(SEPARATOR)
        print()