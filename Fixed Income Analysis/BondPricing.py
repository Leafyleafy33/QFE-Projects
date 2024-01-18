import pandas as pd
import numpy as np
import calendar
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import fsolve


class Bond:
    def __init__(self, maturity, cr, ytm, name="Bond", freq=2, convention=360, current_date=None, settle=None, face_value=100):
        if not (bool(current_date) ^ bool(settle)):
            raise ValueError("Please provide either a current_date or a settle date, not both or neither.")
        
        self.name = name
        self.maturity = pd.Timestamp(maturity)
        self.settle, _ = self._adjusted_date_and_daycount((pd.Timestamp(current_date) + pd.DateOffset(days=2)) if current_date else (pd.Timestamp(settle)), None)
        self.period = convention / freq
        self.cr = cr
        self.pmt = cr * face_value / freq
        self.ytm = ytm
        self.freq = freq
        self.convention = convention
        self.face_value = face_value
        self.Dates = np.array(self._compute_dates()[1])
        self.Coupon_Dates = np.array(self._compute_dates()[0])
        self.A = np.array(self._compute_cashflows())
        self.N = len(self.Dates)
        self.difference = self._days_since_last_coupon()

    def _adjusted_date(self, date):
        date = pd.Timestamp(date)
        holidays = USFederalHolidayCalendar().holidays(start=date, end=self.maturity)
        while date.weekday() >= 5:
                date += pd.DateOffset(days=1)
        while date in holidays:
            date += pd.DateOffset(days=1)
        return date
    
    def _day_count(self, date1, date2):
        d1, m1, y1 = date1.day, date1.month, date1.year
        d2, m2, y2 = date2.day, date2.month, date2.year
        d1 = min(30, d1)
        if d1 == 30:
           d2 = min(d2, 30)
        if m1 == 2 and d1 >= 28:
            if not calendar.isleap(y1) or (calendar.isleap(y1) and d1 == 29):
                d1 = 30
        if m1 == 2 and d1 == 30 and m2 == 2 and d2 >= 28:
            if not calendar.isleap(y2) or (calendar.isleap(y2) and d2 == 29):
                d2 = 30
        days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return days
    
    def _adjusted_date_and_daycount(self, date1, date2):
        date1 = self._adjusted_date(date1)
        date2 = self._adjusted_date(date2) if date2 else None
        days = self._day_count(date1, date2) if date2 else None
        return date1, days

    def _compute_dates(self):
        coupon_dates = [self.maturity]
        offset = int(self.period / 30)
        next_date = coupon_dates[-1] - pd.DateOffset(months=offset)
        while next_date >= (self.settle):
            coupon_dates.append(next_date)
            next_date = next_date - pd.DateOffset(months=offset)
            
        coupon_dates.append(next_date)

        pay_dates = []
        for cdt in coupon_dates:
            adjusted_date, _ = self._adjusted_date_and_daycount(cdt, None)
            pay_dates.append(adjusted_date)

        pay_dates.pop()

        self.Coupon_Dates = np.array(coupon_dates)
        self.Pay_Dates = np.array(pay_dates)

        return self.Coupon_Dates, self.Pay_Dates
    
    def _compute_cashflows(self):
        n_periods = np.array([(self._adjusted_date_and_daycount(self.settle, s)[1] / (self.convention / self.freq)) for s in self.Pay_Dates])
        cf = [self.cr * self.face_value / self.freq] * len(self.Pay_Dates)
        if cf:
            cf[0] += self.face_value
        return list(zip(self.Pay_Dates, n_periods, cf))

    def _days_since_last_coupon(self):
        return self._day_count(self.Coupon_Dates[-1], self.settle)
        
    def accrued_interest(self):
        return ((self.cr * 100) / self.freq) * (self._days_since_last_coupon() / self.period)
    
    def clean_price(self):
        dirty_price, accrued_interest = self.dirty_price(), self.accrued_interest()
        return dirty_price - accrued_interest
    
    def dirty_price(self):
        discounted_cashflows = self.A[:, 2] / (1 + self.ytm / self.freq) ** self.A[:, 1]
        return np.sum(discounted_cashflows)

    def calculate_expected_return(self, reinvestment_rate, future_ytm, selling_period_years):
        years_to_maturity = self.maturity.year - self.settle.year
        buy_price = (self.cr / self.freq) / (self.ytm / self.freq) * (1 - (1 + self.ytm / self.freq)**-(years_to_maturity * self.freq)) + 100 / (1 + self.ytm / self.freq)**(years_to_maturity * self.freq)
        sell_price = (self.cr / self.freq) / (future_ytm / self.freq) * (1 - (1 + future_ytm / self.freq)**-((years_to_maturity * self.freq) - (selling_period_years * self.freq)))+ 100 / (1 + future_ytm / self.freq)**((years_to_maturity * self.freq) - (selling_period_years * self.freq))
        fv_reinvested_coup = (self.cr / self.freq) / (reinvestment_rate / self.freq) * (1 - (1 + reinvestment_rate / self.freq)**-(selling_period_years * self.freq)) * (1 + reinvestment_rate / self.freq)**(selling_period_years * self.freq)
        expected_return = (sell_price - (buy_price * (1 + reinvestment_rate / self.freq)**(selling_period_years * self.freq)) + fv_reinvested_coup) / buy_price * (1 + reinvestment_rate / self.freq)**(selling_period_years * self.freq)
        return {
            "Buy Price": round(buy_price, 7),
            "Sell Price": round(sell_price, 7),
            "Future Value of Reinvested Coupons": round(fv_reinvested_coup, 7),
            "Expected Return": round(expected_return, 7)
        }
    def compare_bonds(bond1, bond2, reinvestment_rate, future_ytm_bond1, future_ytm_bond2, selling_period_years):
        bond1_return = bond1.calculate_expected_return(reinvestment_rate, future_ytm_bond1, selling_period_years)["Expected Return"]
        bond2_return = bond2.calculate_expected_return(reinvestment_rate, future_ytm_bond2, selling_period_years)["Expected Return"]
        print(f"Statstics for {bond1.name} and {bond2.name}:"
                f"\nReinvestment Rate: {reinvestment_rate}"
                f"\nFuture YTM for {bond1.name}: {future_ytm_bond1}"
                f"\nFuture YTM for {bond2.name}: {future_ytm_bond2}"
                f"\nSelling Period: {selling_period_years} years\n")
    
        print(f"{bond1.name} expected return: {bond1_return}")
        print(f"{bond2.name} expected return: {bond2_return}")

        if bond1_return > bond2_return:
            return f"{bond1.name} is the better investment with an expected return of {bond1_return}"
        else:
            return f"{bond2.name} is the better investment with an expected return of {bond2_return}"
       
    def macaulay_duration(self):
        P0 = self.dirty_price()  
        duration = sum(t * cf / ((1 + self.ytm / self.freq) ** t) for _, t, cf in self.A)
        macaulay_duration = (duration / P0) * (1 / self.freq)
        return macaulay_duration
    
    def modified_duration(self):
        D = self.macaulay_duration()
        modified_duration = -D / (1 + self.ytm / self.freq)
        return -modified_duration
    
    def dv01(self):
        duration = -self.duration()
        return duration
    
    def duration(self):
        n_periods = self.A[:, 1]
        cash_flows = self.A[:, 2]
        discount_factors = (1 + self.ytm / self.freq)**(n_periods + 1)
        dP = np.sum(-n_periods * cash_flows / discount_factors) * (1 / self.freq)
        return dP

    def convexity(self):
        cash_flows = self.A[:, 2]
        t_periods = self.A[:, 1]
        discount_rates = (1 + self.ytm / self.freq) ** t_periods
        coefficients = cash_flows * t_periods * (t_periods + 1) / self.dirty_price()
        return np.sum(coefficients / discount_rates) / self.freq ** 2

    def modified_duration_approx(self):
        original_ytm = self.ytm
        self.ytm += 0.000005
        price_high = self.dirty_price()
        
        self.ytm = original_ytm - 0.000005
        price_low = self.dirty_price()
        
        self.ytm = original_ytm
        
        dp_dytm = (price_high - price_low) / 0.00001
        P0 = self.dirty_price()  
        
        D_star_approx = -(dp_dytm / P0)
        return D_star_approx
    
    def coupon_payments_received(self, purchase_date, sale_date):
            # Calculate coupon payments received between purchase and sale dates
        coupon_dates_between = [date for date in self.Coupon_Dates if purchase_date < date <= sale_date]
        coupon_payments = len(coupon_dates_between)
        
        print(f"Coupon payments between purchase and sale dates: {coupon_payments}")
        return coupon_payments
    
    def holding_period_return(self, purchase_date, purchase_price, sale_date, p_sell):
        # Calculate total coupon payments received during the holding period
        coupon_payments_received = self.coupon_payments_received(purchase_date, sale_date) * self.pmt
        print(f"Total coupon payments received during the holding period: {coupon_payments_received:.4f}")
        total_received = p_sell + coupon_payments_received
        
        print(f"Total received from the bond during the holding period: {total_received:.6f}")
        hpr = (total_received - purchase_price) / purchase_price
        
        print(f"Holding period return: {hpr:.2%}")
        return hpr
                
    def display_parameters(self):

        section_width = 29
        
        # Header
        print(" " * 32 + "Bond Details")
        print("-" * 92)
        print()
        
        # General Info
        general_info = {
            "Name": self.name,
            "Face Value": f"${self.face_value:.2f}",
            "Maturity": self.maturity.strftime('%Y-%m-%d'),
            "Settled ": self.settle.strftime('%Y-%m-%d'),
            "Prev Pmt": self.Coupon_Dates[-1].strftime('%Y-%m-%d'),
            "Next Pmt": self.Dates[-1].strftime('%Y-%m-%d'),
        }
        
        # Rates
        rates = {
            "Coupon Rate": f"{self.cr * 100:.3f}%",
            "Yield Rate": f"{self.ytm * 100:.6f}%",
            "Current Yield": f"{self.cr * self.face_value / self.dirty_price():.4%}",
            "Coupon Frequency": f"{self.freq}",
            "Mac Duration": f"{self.macaulay_duration():.6f}",
            "Mod Duration": f"{self.modified_duration():.6f}",

        }
        
        # Periods
        periods = {
            "Period": f"{self.convention / self.freq:.0f} days",
            "Days since coupon": f"{self._days_since_last_coupon()}",
            "N": len(self.Dates),
            "PMT": f"${self.pmt:.4f}",
            "Convexity": f"{self.convexity():.4f}",
            "DV01 Per MM": f"${self.dv01():.6f}",
        }
    
        # Display Segments
        segments = [general_info, rates, periods]
        max_len = max([len(segment) for segment in segments])
        
        for i in range(max_len):
            for segment in segments:
                keys = list(segment.keys())
                if i < len(segment):
                    key = keys[i]
                    value = segment[key]
                    print(f"{key}: {value}".ljust(section_width), end="   ")
                else:
                    print(" " * section_width, end="   ")
            print()
        print()
        
        # Price Info
        prices = {
            "DirtyPrice": f"${self.dirty_price():.7f}",
            "AccruedInt": f"${self.accrued_interest():.7f}",
            "CleanPrice": f"${self.clean_price():.7f}",
        }
        
        max_price_key_len = max([len(key) for key in prices.keys()])
        
        for key, value in prices.items():
            print(f"{key}".rjust(52 - max_price_key_len) + f": {value}")
        print("-" * 92)
    
    
class BondPortfolio:
    def __init__(self):
        self.bonds = []  # List to store Bond objects
        self.face_values = []  # Corresponding face values for each bond

    def add_bond(self, bond, face_value):
        self.bonds.append(bond)
        self.face_values.append(face_value)

    def market_value(self):
        return sum((Bond.dirty_price() /100) * face_value for Bond, face_value in zip(self.bonds, self.face_values))

    def _weight(self, face_value):
        return face_value / sum(self.face_values)

    def portfolio_duration(self):
        return sum(self._weight(face_value) * Bond.modified_duration() for Bond, face_value in zip(self.bonds, self.face_values))

    def portfolio_dv01(self):
        return sum(self._weight(face_value) * Bond.dv01() for Bond, face_value in zip(self.bonds, self.face_values))

    def display_parameters(self):
        header = "Bond Portfolio Details"
        line_length = 85  # Adjusted to reach the end of the 'Weight' column
        print(header.center(line_length))
        print('-' * line_length)

        # Column headers
        col_headers = ["Name", "Coupon Rate", "Yield", "D*", "DV01", "Weight"]

        # Widths for each column (based on max length of data in each column)
        col_widths = [max(len(Bond.name) for Bond in self.bonds) + 2, 12, 10, 10, 10, 10]

        # Displaying bond details in a table-like format
        for header, width in zip(col_headers, col_widths):
            print(header.center(width), end=" | ")
        print()
        print('-' * line_length)

        for Bond, face_value in zip(self.bonds, self.face_values):
            bond_data = [
                Bond.name,
                f"{Bond.cr * 100:.2f}%",
                f"{Bond.ytm * 100:.4f}%",
                f"{Bond.modified_duration():.4f}",
                f"{Bond.dv01():.4f}",
                f"{self._weight(face_value) * 100:.2f}%"
            ]
            for data, width in zip(bond_data, col_widths):
                print(data.ljust(width), end=" | ")
            print()

        print('=' * line_length)

        # Displaying portfolio details (centered but left-aligned)
        print("Portfolio Metrics".center(line_length))
        print(f"\tMarket Value: \t${self.market_value()/1e6:,.4f} Million".center(line_length))  # Displaying in millions
        print(f"Portfolio D*: \t{self.portfolio_duration():.4f}".center(line_length))
        print(f"Portfolio DV01: \t${self.portfolio_dv01():,.4f}".center(line_length))
        print('=' * line_length)


class YieldCurveTrade:
    def __init__(self, bond_short, bond_long1, bond_long2):
        self.bond_short = bond_short
        self.bond_long1 = bond_long1
        self.bond_long2 = bond_long2
        self.w1 = 0
        self.w2 = 0

        self.original_dirty_price_short = bond_short.dirty_price()
        self.original_dirty_price_long1 = bond_long1.dirty_price()
        self.original_dirty_price_long2 = bond_long2.dirty_price()

    def calculate_weights(self):
        a = self.bond_short.dirty_price() / self.bond_long1.dirty_price()
        b = self.bond_long2.dirty_price() / self.bond_long1.dirty_price()

        c = self.bond_long1.dv01()
        d = self.bond_long2.dv01()
        e = self.bond_short.dv01()

        self.w2 = (e - (a * c)) / (d - (b * c))
        self.w1 = (a - self.w2 * b)
        
        return self.w1, self.w2

    def pnl_from_shift(self, shift_short, shift_long1, shift_long2):
        old_short_price = self.original_dirty_price_short
        old_long1_price = self.original_dirty_price_long1
        old_long2_price = self.original_dirty_price_long2

        # Adjust the yields temporarily
        self.bond_short.ytm += shift_short
        self.bond_long1.ytm += shift_long1
        self.bond_long2.ytm += shift_long2

        new_short_price = self.bond_short.dirty_price()
        new_long1_price = self.bond_long1.dirty_price()
        new_long2_price = self.bond_long2.dirty_price()

        profit = self.w1 * (new_long1_price - old_long1_price) + \
                 self.w2 * (new_long2_price - old_long2_price) - \
                 (new_short_price - old_short_price)
        
        print(f"Profit from shift: {self.w1:.6f} * ({new_long1_price:.6f} - {old_long1_price:.4f}) + w2{self.w2:.4f} * ({new_long2_price:.4f} - {old_long2_price:.4f}) - ({new_short_price:.4f} - {old_short_price:.4f}) = {profit:.4f}")
        
        # Reset the yields
        self.bond_short.ytm -= shift_short
        self.bond_long1.ytm -= shift_long1
        self.bond_long2.ytm -= shift_long2
        
        return profit

class Trade:
    def __init__(self, bond1, bond2, spread_increase):
        self.bond1 = bond1
        self.bond2 = bond2
        self.spread_increase = spread_increase  # True for increase, False for decrease

    def buy_or_sell(self):
        if self.spread_increase:
            # If spread is expected to increase
            if self.bond1.ytm < self.bond2.ytm:
                return f"Short-sell {self.bond1.name} and Buy (go long) {self.bond2.name}"
            else:
                return f"Short-sell {self.bond2.name} and Buy (go long) {self.bond1.name}"
        else:
            # If spread is expected to decrease
            if self.bond1.ytm < self.bond2.ytm:
                return f"Buy (go long) {self.bond1.name} and Short-sell {self.bond2.name}"
            else:
                return f"Buy (go long) {self.bond2.name} and Short-sell {self.bond1.name}"

    def inflation_expectation(self, spread_direction):
        if spread_direction == "increase":
            # If spread is expected to increase
            return "Two economic reasons for an increasing spread might be:\n1. Expectations of rising inflation in the future.\n2. An anticipated increase in short-term interest rates by the central bank."
        else:
            # If spread is expected to decrease
            return "Two economic reasons for a decreasing spread might be:\n1. Expectations of stable or falling inflation.\n2. An anticipated monetary easing by the central bank."

    def amount_to_buy(self, short_sell_amount):
        if self.spread_increase:
            # Reverse the calculation if spread is expected to increase
            if self.bond1.ytm > self.bond2.ytm:
                n_long = (short_sell_amount * self.bond1.dv01()) / self.bond2.dv01()
            else:
                n_long = (short_sell_amount * self.bond2.dv01()) / self.bond1.dv01()
        else:
            if self.bond1.ytm < self.bond2.ytm:
                n_long = (short_sell_amount * self.bond1.dv01()) / self.bond2.dv01()
            else:
                n_long = (short_sell_amount * self.bond2.dv01()) / self.bond1.dv01()
        return n_long

    def market_value_positions(self, short_sell_amount, n_long):
        if self.spread_increase:
            if self.bond1.ytm > self.bond2.ytm:
                market_value_short = short_sell_amount * self.bond2.dirty_price()
                market_value_long = n_long * self.bond1.dirty_price()
            else:
                market_value_short = short_sell_amount * self.bond1.dirty_price()
                market_value_long = n_long * self.bond2.dirty_price()
        else:
            if self.bond1.ytm < self.bond2.ytm:
                market_value_short = short_sell_amount * self.bond2.dirty_price()
                market_value_long = n_long * self.bond1.dirty_price()
            else:
                market_value_short = short_sell_amount * self.bond1.dirty_price()
                market_value_long = n_long * self.bond2.dirty_price()
        return market_value_short, market_value_long

    def profit_or_loss(self, short_sell_amount, n_long, bp_change_1, bp_change_2):
        original_dirty_price_1 = self.bond1.dirty_price()
        original_dirty_price_2 = self.bond2.dirty_price()
        
        self.bond1.ytm += bp_change_1
        self.bond2.ytm += bp_change_2

        profit_loss_bond2 = (self.bond2.dirty_price() - original_dirty_price_2)/100 * n_long
        profit_loss_bond1 = (self.bond1.dirty_price() - original_dirty_price_1)/100 * short_sell_amount
        return profit_loss_bond1, profit_loss_bond2
    
    def display(self, short_sell_amount, bp_change_1, bp_change_2):
        # (a) Compute the DV01 of each bond
        dv01_1 = self.bond1.dv01()
        dv01_2 = self.bond2.dv01()
        print(f"(a) DV01 of {self.bond1.name}: ${dv01_1:,.2f}")
        print(f"    DV01 of {self.bond2.name}: ${dv01_2:,.2f}\n")

        # (b) Construct a trade
        print("(b) " + self.buy_or_sell())
        n_long = self.amount_to_buy(short_sell_amount)
        n_long = n_long/1000000
        short_sell_amount = short_sell_amount/1000000
        print(f"    Short-sell ${short_sell_amount:.2f} MM of {self.bond1.name if self.bond1.ytm < self.bond2.ytm else self.bond2.name} and buy ${n_long:.4f} MM of {self.bond2.name if self.bond1.ytm < self.bond2.ytm else self.bond1.name} to balance the DV01 exposure.\n")
        
        # New profit or loss calculation with exact values
        profit_loss_1, profit_loss_2 = self.profit_or_loss(short_sell_amount, n_long, bp_change_1, bp_change_2)[0], self.profit_or_loss(short_sell_amount, n_long, bp_change_1, bp_change_2)[1]
        print(f"(c) Exact profit or loss after YTM changes: Bond1 = ${profit_loss_1:,.6f}, Bond2 = ${profit_loss_2:,.6f}\n")
        pnl = profit_loss_2 - profit_loss_1
        print(f"Gain/loss in Dollars: ${pnl*1000000:,.2f}\n")
        # (d) Determine the spread's direction and compute the gain or loss
        original_spread = self.bond1.ytm - bp_change_1 - (self.bond2.ytm - bp_change_2)
        new_spread = self.bond1.ytm - self.bond2.ytm
        spread_direction = "decreased" if new_spread > original_spread else "increased"
        print(f"(d) The spread has {spread_direction}.\n")
        
        # Loss calculation
        loss = profit_loss_2 - profit_loss_1
        print(f"Loss due to spread change: ${loss*1000000:,.2f}\n")

        print("Explanation:")
        print(self.inflation_expectation(spread_direction))