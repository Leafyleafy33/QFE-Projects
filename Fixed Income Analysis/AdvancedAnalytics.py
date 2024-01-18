import numpy as np
import numpy_financial as npf
import pandas as pd
import sympy as sp
import pandas_datareader as pdr
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from scipy.optimize import minimize, root_scalar
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm
from BondPricing import Bond
import yfinance as yf
from datetime import datetime, timedelta

# Customize Seaborn style
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
sns.set_palette("deep")

class NelsonSiegelModel:
    def __init__(self, maturity_dict, date, type='Daily'):
        self.maturity_dict = maturity_dict
        self.type = type
        self.datestring = date
        self.date = pd.to_datetime(date)
        self.treasuries = self.fetch_bond_data()
        self.lambda_vals = np.linspace(0.01, 1, 100)
        self.precomputed_terms = self.precompute_terms()
        self.NS_df = self.calculate_NS_params()
        self.Z = self.calculate_Z()

    def calculate_Z_for_dates(self, specific_dates):
        Z_mat = np.array([(d - self.date).days / 360 for d in specific_dates])
        Z = pd.DataFrame({'Rate': self.NS(self.NS_df.iloc[0], Z_mat), 'Maturity': Z_mat, 'Date': specific_dates})
        return Z
    
    def fetch_bond_data(self):
        bonds = pdr.get_data_fred(self.maturity_dict.keys(), start=self.datestring, end=self.datestring)
        bonds = bonds.reset_index().melt(id_vars='DATE', var_name='Bond', value_name='Yield')
        bonds['Maturity'] = bonds['Bond'].map(self.maturity_dict)
        return bonds

    def precompute_terms(self):
        maturity = self.treasuries['Maturity'].values
        return {'exp_term': np.exp(-np.outer(self.lambda_vals, maturity)),
                'maturity_term': (1 - np.exp(-np.outer(self.lambda_vals, maturity))) / np.outer(self.lambda_vals, maturity)}

    def NS_eq(self, c, maturity, yield_data, L, precomputed_terms):
        idx = np.argmin(np.abs(L - self.lambda_vals))
        exp_term = precomputed_terms['exp_term'][idx]
        maturity_term = precomputed_terms['maturity_term'][idx]

        ns = c[0] + c[1] * maturity_term + c[2] * (maturity_term - exp_term)
        e = yield_data - ns
        return np.dot(e, e), -2 * np.array([e.sum(), np.dot(e, maturity_term), np.dot(e, maturity_term - exp_term)])
    
    def NS_params(self, maturity, yield_data):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(minimize, self.NS_eq, [0.01, 0.01, 0.01], args=(maturity, yield_data, l, self.precomputed_terms), method='L-BFGS-B', jac=True) for l in self.lambda_vals]
            results = [f.result() for f in futures]
        min_index = np.argmin([res.fun for res in results])
        return np.append(results[min_index].x, self.lambda_vals[min_index])

    def calculate_NS_params(self):
        group = self.treasuries.groupby('DATE').get_group(self.date)
        return pd.DataFrame([self.NS_params(group['Maturity'].values, group['Yield'].values)], columns=['beta0', 'beta1', 'beta2', 'lambda'])
   
    @staticmethod
    def NS(df, M):
        lambda_M = df['lambda'] * M
        exp_lambda_M = np.exp(-lambda_M)
        return df['beta0'] + df['beta1'] * (1 - exp_lambda_M) / lambda_M + df['beta2'] * ((1 - exp_lambda_M) / lambda_M - exp_lambda_M)
    
    def calculate_Z(self):
        type_lower = self.type.lower()

        if type_lower in ['daily', 'day', 'per day', 'each day', 'day-to-day', 'every day', 'everyday', 'dy', 'd']:
            Z_mat = np.arange(1/12/30, 30+(1/12/30), 1/12/30)
        elif type_lower in ['monthly', 'month', 'per month', 'each month', 'month-to-month', 'every month', 'mo', 'mth', 'm']:
            Z_mat = np.arange(1/12, 30+(1/12), 1/12)
        elif type_lower in ['quarterly', 'quarter', 'per quarter', 'each quarter', 'quarter-to-quarter', 'every quarter', 'qtr', 'qrtr', 'q']:
            Z_mat = np.arange(1/4, 30+(1/4), 1/4)
        elif type_lower in ['semi-annually', 'semi-an', 'semi', 'semi-annual', 'semi-ann', 'biannual', 'bi-annually', 'half-yearly', 'twice a year', 'sa', 's']:
            Z_mat = np.arange(.5, 30+.5, .5)
        elif type_lower in ['annually', 'yearly', 'annual', 'year', 'per annum', 'each year', 'year-to-year', 'every year', 'yr', 'y', 'a']:
            Z_mat = np.arange(1, 30+1, 1)
        else:
            raise ValueError('Invalid type. Please choose one of the following: Daily, Monthly, Quarterly, Semi-Annually, Annually')

        dates = [self.date + pd.Timedelta(days=360 * m) for m in Z_mat]
        Z = pd.DataFrame({'Rate': self.NS(self.NS_df.iloc[0], Z_mat), 'Maturity': Z_mat, 'Date': dates})
        return Z

    
    def plot_NS_rates(self, show=False):
        plt.plot(self.Z['Maturity'], self.Z['Rate'])
        plt.title('Nelson-Siegel rates')
        plt.xlabel('Maturity')
        plt.ylabel('Rate')
        plt.show()
        if show:
            display(self.Z)

class BondAnalytics:
    def __init__(self, tickers, date):
        self.nelson_siegel_model = NelsonSiegelModel(tickers, date)
        self.zero_curve = self.nelson_siegel_model.Z
        
    def calculate_payment_dates(self, bond):
        # Calculate exact payment dates
        return [bond.settle + np.timedelta64(int(i * 6), 'M') for i in range(1, bond.N + 1)]

    def find_corresponding_zero_rates(self, payment_dates):
        if 'Date' not in self.nelson_siegel_model.Z.columns:
            raise KeyError("The 'Date' column is missing in the zero curve DataFrame.")
        
        # Match each payment date with the closest date in the zero curve
        matched_rates = []
        for date in payment_dates:
            closest_date = min(self.zero_curve['Date'], key=lambda d: abs(d - date))
            matched_rate = self.zero_curve[self.zero_curve['Date'] == closest_date]['Rate'].iloc[0]
            matched_rates.append(matched_rate)
        return np.array(matched_rates)

    
    def calculate_z_spread(self, bond):
        payment_dates = self.calculate_payment_dates(bond)
        z_rates = self.find_corresponding_zero_rates(payment_dates)
        z_rates /= 100

        def present_value(rp):
            discount_factors = 1 / (1 + z_rates/2 + rp/2)**(np.arange(1, bond.N + 1))
            cash_flows = np.full(bond.N, bond.cr * bond.face_value / 2)
            cash_flows[-1] += bond.face_value
            return np.sum(cash_flows * discount_factors) - bond.dirty_price()
        
        bracket_range = [0, 5]  # Adjust this range as needed

        # Check if the function crosses zero within the range
        if present_value(bracket_range[0]) * present_value(bracket_range[1]) > 0:
            print(f"Cannot find a root within the bracket {bracket_range} for bond {bond.name}.")
            return None

        result = root_scalar(present_value, bracket=bracket_range, method='brentq')
        return result.root * 100
    

class BlackScholesModel:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, parse_dates=['date'])
        self.myTol = 1e-8

    def myBS(self, V, row, sA, sE):
        E, F, r, t = row['Ve'], row['F_debt'], row['Tbill'] / 100, 1
        d1 = (np.log(V/F) + (r + 0.5 * sA**2) * t) / (sA * np.sqrt(t))
        d2 = d1 - sA * np.sqrt(t)
        
        e1 = E - (V * norm.cdf(d1) - F * np.exp(-r * t) * norm.cdf(d2))
        e2 = sE - (V / E) * norm.cdf(d1) * sA
        
        return e1**2 + e2**2

    def minBS(self, df, sA, sE):
        results = np.zeros(len(df))

        for i in range(len(df)):
            result = minimize(self.myBS, x0=df.iloc[i]['V0'], args=(df.iloc[i], sA, sE))
            results[i] = result.x

        return results

    def iterate_calculations(self, df):
        df_local = df.copy()
        df_local['Ve'] = df_local['stock_price'] * df_local['shares']
        df_local['ret'] = df_local['stock_price'].pct_change()
        sE = sA = np.sqrt(12) * df_local['ret'].std()
        df_local['V0'] = df_local['Ve'] + df_local['F_debt']

        for i in range(20):
            df_local['Va'] = self.minBS(df_local, sA, sE)
            sA_new = np.sqrt(12) * df_local['Va'].pct_change().std()
            
            if np.abs(sA - sA_new) < self.myTol:
                break
            sA = sA_new
            df_local['V0'] = df_local['Va']

        df_local['DtD'] = (df_local['Va'] - df_local['F_debt']) / (sA * df_local['Va'])
        df_local['EDF'] = norm.cdf(-df_local['DtD'])
        return df_local

    def plot_EDF(self):
        fig, ax = plt.subplots(figsize=(8.3, 10*0.618), facecolor='#2c2c2c')

        for ticker in self.df['ticker'].unique():
            df_ticker = self.df[self.df['ticker'] == ticker].copy()
            df_ticker = self.iterate_calculations(df_ticker)
            ax.plot(df_ticker['date'], df_ticker['EDF'], label=f'EDF - {ticker}')

        ax.set_xlabel('Date')
        ax.set_ylabel('EDF Probability')
        ax.set_title('Expected Default Frequency (EDF) Over Time for Companies')
        # Customize the plot
        ax.set_xlabel('Time')
        ax.set_ylabel('EDF (%)')
        ax.set_title('Expected Default Frequency (EDF) Over Time for Companies')

        # Customize the style
        for spine in ax.spines.values():
            spine.set_color("#66c2a5")
        ax.set_facecolor("#2c2c2c")

        # Add legend, grid, and adjust layout
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()

class BondInterpolator:
    def __init__(self, maturities, prices, coupon_rates, ytms, target_maturities=None):
        self.maturities = maturities
        self.prices = prices
        self.coupon_rates = coupon_rates
        self.ytms = ytms
        self.target_maturities = target_maturities if target_maturities else 0
        self.all_bonds = {}
        self.A = None
        self.table = None
        
    def interpolate(self):
        all_maturities = np.sort(self.maturities + self.target_maturities)
        self.all_bonds = {
            'Maturity': all_maturities,
            'Price': np.interp(all_maturities, self.maturities, self.prices),
            'Coupon (%)': np.interp(all_maturities, self.maturities, self.coupon_rates),
            'YTM (%)': np.interp(all_maturities, self.maturities, self.ytms)
        }
        
    def generate_cash_flow_matrix(self):
        Maturity = self.all_bonds['Maturity']
        Coupon_Rate = self.all_bonds['Coupon (%)']
        
        self.A = np.zeros((len(Maturity), len(Maturity)))
        np.fill_diagonal(self.A, 100 + Coupon_Rate)
        for i in range(1, len(Maturity)):
            self.A[i, :i] = Coupon_Rate[i]
            
    def calculate_zero_prices(self):
        Price = self.all_bonds['Price']
        Z = np.linalg.solve(self.A, Price)
        
        self.table = pd.DataFrame({'Maturity': self.all_bonds['Maturity'], 'Z.prc': Z})
        self.table['z.rate'] = (1 / self.table['Z.prc'])**(1 / self.table['Maturity']) - 1
        self.table['FV_Test'] = npf.fv(self.table['z.rate'], self.table['Maturity'], 0, -self.table['Z.prc'])
        self.table[['Z.prc', 'z.rate', 'FV_Test']] = self.table[['Z.prc', 'z.rate', 'FV_Test']].multiply([100, 100, 100])
        
    def bootstrap_z_rates(self):
        z_rates = []
        for i, maturity in enumerate(self.all_bonds['Maturity']):
            if self.all_bonds['Coupon (%)'][i] == 0:
                # For zero-coupon bonds, YTM is the zero rate
                z_rates.append(self.all_bonds['YTM (%)'][i] / 100)
            else:
                cashflows = [self.all_bonds['Coupon (%)'][i]]
                for _ in range(int(maturity) - 1):
                    cashflows.append(self.all_bonds['Coupon (%)'][i])
                cashflows[-1] += 100  # Adds face value to the last cash flow
                
                sum_previous = sum([cf / (1 + z_rates[j])**(j+1) for j, cf in enumerate(cashflows[:i])])
                z = ((self.all_bonds['Price'][i] - sum_previous) / cashflows[i]) ** (-1/(maturity)) - 1
                z_rates.append(z)
        self.all_bonds['Bootstrap Z Rates (%)'] = np.array(z_rates) * 100

    
    def symbolic_bootstrap(self):
        # Definitions
        m = sp.symbols('m', integer=True)
        z = sp.symbols('z:%d' % (max(self.all_bonds['Maturity']) + 1))
        p = sp.symbols('p:%d' % len(self.all_bonds['Maturity']))
        c = self.all_bonds['Coupon (%)']

        # Initial formulas list
        formulas = []
        
        # For zero-coupon bonds
        zero_formula = sp.Eq(p[0], 100 / (1 + z[1]))

        # For coupon bonds
        for i, maturity in enumerate(self.all_bonds['Maturity']):
            cashflows = [c[i]]
            for _ in range(int(maturity) - 1):
                cashflows.append(c[i])
            cashflows[-1] += 100
            current_formula = p[i]
            for j, cf in enumerate(cashflows):
                current_formula -= cf / (1 + z[j+1])**(j+1)
            formulas.append(sp.Eq(0, current_formula))

        return zero_formula, formulas

    def display_bootstrap(self):
        zero_formula, formulas = self.symbolic_bootstrap()

        print("- We can back out the implied zero rates by noting that")
        display(zero_formula)
        for formula in formulas:
            display(formula)

        print("\n- We can solve for the z_i's recursively, hence the bootstrap name.")
        for i, z_rate in enumerate(self.all_bonds['Bootstrap Z Rates (%)']):
            print(f"  - z_{i+1} = {z_rate:.2f}%")
        
    def run(self):
        self.interpolate()
        self.generate_cash_flow_matrix()
        self.calculate_zero_prices()
        self.bootstrap_z_rates()
        
        # Display results
        all_bonds_df = pd.DataFrame(self.all_bonds).style.set_caption('All Interpolated Bonds').format("{:.2f}").hide()
        A_df = pd.DataFrame(self.A).style.set_caption('Cash Flow Matrix (A)').format("{:.2f}").hide()
        table = self.table.style.set_caption('Zero Rates Matrix').format("{:.2f}").hide()
        
        display(all_bonds_df, A_df, table)


class TreasuryYieldAnalysis:
    def __init__(self, NS_model):
        self.NS_model = NS_model
        self.zero_rates = NS_model.Z['Rate'] / 100  # Zero rates from the Nelson-Siegel model
        self.maturities = NS_model.Z['Maturity']    # Maturities from the Nelson-Siegel model

    def compute_forward_rate(self, start_year, end_year):
        # Extract zero rates for the specified years
        z1 = self.zero_rates[np.isclose(self.maturities, start_year)].iloc[0]
        z2 = self.zero_rates[np.isclose(self.maturities, end_year)].iloc[0]

        # Calculate the forward rate
        forward_rate = ((1 + z2)**end_year / (1 + z1)**start_year)**(1 / (end_year - start_year)) - 1
        return forward_rate
    
    def display_forward_rates_markdown(self, start_year, end_year):
        forward_rate = self.compute_forward_rate(start_year, end_year)
        markdown_output = f"$f_{{{start_year}\\rightarrow{end_year}}}$ = {forward_rate:.2%}"
        return markdown_output
    
    def market_expectation(self, forward_rate_1_2, forward_rate_2_3):
        # Compute the current 1-year zero rate
        z1 = self.zero_rates[np.isclose(self.maturities, 1)].iloc[0]
        # Determine market expectations
        expectation1 = "**_increase_**" if forward_rate_1_2 > z1 else "**_decrease_**"
        expectation2 = "**_increase_**" if forward_rate_2_3 > z1 else "**_decrease_**"
        
        # Create a markdown output summarizing the expectations
        markdown_output = f"(b) Given the forward rates, The market expects the 1-year interest rate to {expectation1} in the next year and to {expectation2} in the following year."
        return markdown_output
    
    def investment_decision_markdown(self, investment_horizon, expected_rate_year1):
        # Current spot rates
        z1 = self.zero_rates[np.isclose(self.maturities, 1)].iloc[0]
        z2 = self.zero_rates[np.isclose(self.maturities, investment_horizon)].iloc[0]

        # Total return from investing in a bond for the investment horizon
        total_return_bond = ((1 + z2)**investment_horizon) - 1

        # Total return from rolling over bonds
        total_return_roll = (1 + z1) * (1 + expected_rate_year1) - 1

        # Decision based on which total return is greater
        decision = "**investing in a single bond**" if total_return_bond > total_return_roll else "**rolling over bonds**"

        # Formatting the result as Markdown
        markdown_output = f"""
## Investment Decision for {investment_horizon}-Year Horizon
- Current 1-year spot rate: **{z1:.2%}**
- Current {investment_horizon}-year spot rate: **{z2:.2%}**
- Expected $f_{{1\\rightarrow{investment_horizon}}}$: **{expected_rate_year1:.2%}**

- Formula for investing in a {investment_horizon}-year bond:
  - Total Return = (($1 + z_{investment_horizon})^{investment_horizon}) - 1$
  - Where $z_{investment_horizon}$ is the current {investment_horizon}-year spot rate.
  - Total Return: **{total_return_bond:.2%}**
  
- Formula for rolling over one-year bonds:
  - Total Return for $\\text{{First Year}} = 1 + z_1$
  - Total Return for Second Year = $(1 + z_1) \\times (1 + E[z_1]) - 1$
  - Where $z_1$ is the current 1-year spot rate and $E[z_1]$ is the expected 1-year spot rate for the second year.
  - Total Return: **{total_return_roll:.2%}**

Based on the total returns, {decision} is better.
"""
        return markdown_output
    
    def compute_NTFS_markdown(self, start_year, end_year, tbill_maturity):
        # Compute spot rates for the given periods
        z_start = self.zero_rates[np.isclose(self.maturities, start_year)].iloc[0]
        z_end = self.zero_rates[np.isclose(self.maturities, end_year)].iloc[0]

        tbill_rate = self.NS_model.fetch_bond_data()
        tbill_rate = tbill_rate[tbill_rate['Maturity'] == tbill_maturity]['Yield'].iloc[0] / 100

        # Use above method
        forward_rate = self.compute_forward_rate(start_year, end_year)
        # Compute the NTFS
        NTFS = forward_rate - tbill_rate

        # Formatting the result as Markdown
        markdown_output = f"""
## Near-Term Forward Spread (NTFS) Calculation

- **Input Rates**:
  - Spot rate for {start_year} years $z_{{{start_year}}}$: **{z_start:.2%}**
  - Spot rate for {end_year} years $z_{{{end_year}}}$: **{z_end:.2%}**
  - 3-month T-bill rate $z_{{{tbill_maturity}}}$: **{tbill_rate:.2%}**

- **Forward Rate Calculation**:
  - Formula: $f_{{{start_year}\\rightarrow{end_year}}} = \left(\\frac{{(1 + z_{{{end_year}}})^{{{end_year}}}}}{{(1 + z_{{{start_year}}})^{{{start_year}}}}}\\right)^{{\\frac{1}{{end - start}}}} - 1$
  - Calculated Forward Rate: **{forward_rate:.4%}**

- **NTFS Calculation**:
  - Formula: $NTFS = f_{{{start_year}\\rightarrow{end_year}}} - z_{{{tbill_maturity}}}$
  - NTFS Value: **{NTFS:.6%}**

"""
        return markdown_output
    

class RepoTransactionAnalysis:
    def __init__(self, bond, repo_rate, haircut, transaction_type):
        self.bond = bond
        self.repo_rate = repo_rate
        self.haircut = haircut
        self.transaction_type = transaction_type
        self.og_settle = bond.settle
        self.initial_price = bond.dirty_price()

    def compute_profit_loss(self, new_ytm, settlement_date):
        initial_price = self.initial_price
        # Update the YTM and calculate the final bond price after YTM change
        self.bond.ytm = new_ytm
        self.bond.settle = settlement_date
        final_price = self.bond.dirty_price()
        # Calculate lent amount and amount to receive
        lent_amount = initial_price * (1 - self.haircut)
        days_between = (settlement_date - self.og_settle).days
        if self.transaction_type == 'repo':
            interest_from_repo = -lent_amount * (self.repo_rate * days_between / 360)
            interest_from_reverse_repo = 0
        else:
            interest_from_repo = 0
            interest_from_reverse_repo = lent_amount * (self.repo_rate * days_between / 360)

        # Calculate profit or loss based on the transaction type
        if self.transaction_type == 'repo':
            profit_loss = (final_price - initial_price) + interest_from_repo
        elif self.transaction_type == 'reverse_repo':
            profit_loss = (initial_price - final_price) + interest_from_reverse_repo

        profit_loss_percent = (profit_loss / lent_amount) * 100
        
        # Calculate break-even prices
        if self.transaction_type == 'repo':
            break_even_price = initial_price + interest_from_repo
            interest_label = 'Interest for Repo Rate'
        else:
            break_even_price = initial_price + interest_from_reverse_repo
            interest_label = 'Interest for Reverse Repo Rate'

        return profit_loss, profit_loss_percent, initial_price, final_price, lent_amount, interest_from_repo, interest_from_reverse_repo, days_between, break_even_price, interest_label
    
    def display_transaction_type(self):
        display(Markdown(f"### (a) Transaction Type\n- **{self.transaction_type.capitalize()}**"))

    def display_profit_loss(self, new_ytm, settlement_date):
        (profit_loss_dollars, profit_loss_percent, initial_price, final_price, lent_amount, interest_from_repo, interest_from_reverse_repo,
         days_between, break_even_price, interest_label) = self.compute_profit_loss(new_ytm, settlement_date)

        markdown_output = f"""
## {self.transaction_type.capitalize()} Profit/Loss Calculation
- Initial Bond Price (Before YTM Change): **${initial_price:.4f}**
- Updated YTM for Bond Buyback: **{new_ytm:.4%}**
- Final Bond Price (After YTM Change): **${final_price:.4f}**
- Haircut Rate: **{self.haircut:.2%}**
- Lent Amount (after haircut): **${lent_amount:.4f}**
- Days Between Transactions: **{days_between} days**
- Repo Rate: **{self.repo_rate:.2%}**
- {interest_label}: **${(interest_from_repo) if self.transaction_type == 'repo' else abs(interest_from_reverse_repo):.5f}**
- Total Profit/Loss in Dollars: **${profit_loss_dollars:.6f}**
- Total Profit/Loss as % of Lent Capital: **{profit_loss_percent:.4%}**

## Break-even Bond Price
- Initial Bond Price: **${initial_price:.4f}**
- {interest_label}: **${(interest_from_repo) if self.transaction_type == 'repo' else abs(interest_from_reverse_repo):.5f}**
- Break-even Price for {self.transaction_type.capitalize()}: **${break_even_price:.5f}**
"""
        display(Markdown(markdown_output))