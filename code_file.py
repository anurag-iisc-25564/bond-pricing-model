#=======================================================================================
#                          imports
#=======================================================================================

import pandas as pd                                   # for data analysis and handling

import numpy as np                                    # for data manipulation

import matplotlib.pyplot as plt                       # for plotting

from scipy.interpolate import CubicSpline             # for generating the CubicSpline
from scipy.optimize import least_squares              # for evaluating our final model

print("Importing necessary libraries.......")
print("Import completed succesfully !")



#=======================================================================================
#                                    DATA IMPORTS
#=======================================================================================
print("Loading gsec data.........")
gsec_df= pd.read_csv("./data/gsec_18Sep2026(G-Sec).csv",
                         skiprows=4)
print("gsec data loaded successfully !")


print("Loading zcyc data.........")
zcyc_df = pd.read_csv("./data/strips_18Sep2026(ZCYC).csv",
                      skiprows=4)
print("zcyc data loaded successfully !")


print("Loading sdl data.........")
sdl_df =  pd.read_csv("./data/sdl_18Sep2026(SDL).csv",
                      skiprows=4)
print("sdl data loaded successfully !")

print("Loading sdlzcyc data.........")
sdlzcyc_df =  pd.read_csv("./data/sdlzcyc_18Sep2026(SDL_ZCYC).csv",
                      skiprows=4)
print("sdlzcyc data loaded successfully !")



#=========================================================================================
#                   GENERATING TENOR COLUMN and NODAL_BONDS DATASET
#=========================================================================================
print("\n---------Generating Tenor for the gsec dataset-----------\n")
print("Tenor is the length of time from today to the date of the date of maturity")
# 1. Set the Valuation Date (the date of the FBIL file)
VALUATION_DATE = pd.to_datetime('2026-09-18')
print("Creating a new column named Tenor in the gsec_df dataset................")
 # 2. Clean column names and compute Tenor (Time to Maturity in years)
gsec_df['Maturity_dt'] = pd.to_datetime(gsec_df['Maturity(dd-mmm-yyyy)'])
gsec_df['Tenor'] = (gsec_df['Maturity_dt'] - VALUATION_DATE).dt.days / 365.25


print("\nThe target Tenors are [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 40.0]\n")
print("\nFiltering bonds according to target tenors.............." )
target_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 40.0]
selected_indices = []    # empty array to store the indices of desired rows
for t in target_tenors:
    # Find the index of the bond with the minimum tenor difference
    closest_idx = (gsec_df['Tenor'] - t).abs().idxmin()
    selected_indices.append(closest_idx)

print("\nFiltering successful !")
print(f"\nThe selected indices as {selected_indices}")
print("\ngenerated a new dataset of the filtered bonds titled as nodal_bonds")
nodal_bonds = gsec_df.loc[list(dict.fromkeys(selected_indices))].copy().reset_index(drop=True)


print(f"Selected {len(nodal_bonds)} benchmark bonds across the maturity spectrum:")
print(nodal_bonds[['ISIN', 'Coupon', 'Maturity(dd-mmm-yyyy)', 'Price(Rs)', 'YTM% p.a. (Semi-Annual)', 'Tenor']])




#===========================================================================================
#           Creating a function to get all the future cashflows
#===========================================================================================
print("\nCreating a Function (get_bond_cashflows) to generate all the future cashflows for a particular desired bond...........")
def get_bond_cashflows(coupon, maturity_date, val_date=VALUATION_DATE):
        """
        Generates future semi-annual coupon dates and cash flow amounts.
        Face value = 100. Coupons are paid semi-annually (every 6 months).
        """
        coupon_pmt = coupon / 2.0  # Semi-annual interest payment per Rs 100

        # Generate coupon payment dates stepping backwards every 6 months from maturity
        cashflow_dates = []
        curr_date = maturity_date
        while curr_date > val_date:
            cashflow_dates.append(curr_date)
            curr_date = curr_date - pd.DateOffset(months=6)

        # Sort dates chronologically from nearest to farthest
        cashflow_dates = sorted(cashflow_dates)

        # Calculate time fraction (in years) from valuation date for each payment
        times = np.array([(d - val_date).days / 365.25 for d in cashflow_dates])

        # All intermediate payments are coupon_pmt; the last payment includes the Rs 100 principal
        amounts = np.full(len(cashflow_dates), coupon_pmt)
        amounts[-1] += 100.0  # Principal repayment at maturity

        return times, amounts

print("Function succesfully generated !")




#=======================================================================================================
#          Creating a function to get the theoritical PV of bond from the zero-rate curves
#======================================================================================================
print("\nCreating a Function (price_bond) to generate PV of the bond from the zero-rate curves...........")

def price_bond(times, amounts, spline_curve):
        """
        Calculates the theoretical price of a bond using zero rates from the spline.
        times: array of payment times (in years)
        amounts: array of payment amounts (Rs)
        spline_curve: a CubicSpline object that returns zero rate in % for any time t
        """
        # 1. Get the zero rate z(t) from our spline for each payment date
        zero_rates = spline_curve(times)

        # 2. Convert zero rates (%) into Discount Factors using semi-annual compounding
        # Formula: D(t) = 1 / (1 + z/200)^(2*t)
        discount_factors = 1.0 / np.power(1.0 + (zero_rates / 200.0), 2.0 * times)

        # 3. Model Price = sum of each cash flow multiplied by its discount factor
        model_price = np.sum(amounts * discount_factors)
        return model_price
print("Function succesfully generated !")


#======================================================================================================
#                          Executing the model for our nodal bonds
#======================================================================================================
print("Executing the model for the nodal bonds")
# 1. Precompute cash flows for all our selected benchmark (nodal) bonds
nodal_cashflows = []
for idx, row in nodal_bonds.iterrows():
        times, amounts = get_bond_cashflows(row['Coupon'], row['Maturity_dt'])
        nodal_cashflows.append({
            'times': times,
            'amounts': amounts,
            'market_price': row['Price(Rs)'],
            'tenor': row['Tenor']
        })

# 2. Define the knot locations (x-axis: the maturities of our 12 milestone bonds)
knot_tenors = np.array([bond['tenor'] for bond in nodal_cashflows])

# Initial guess for the zero rates (y-axis): we can use the bond's YTM from the CSV
initial_guess = nodal_bonds['YTM% p.a. (Semi-Annual)'].values

# 3. Define the Error Function that Python will try to minimize to 0
def objective_function(guess_rates):
        # Bend a smooth Cubic Spline through the current guessed rates
        current_spline = CubicSpline(knot_tenors, guess_rates, bc_type='natural')

        # Calculate price difference (Error = Model Price - Market Price) for each bond
        errors = []
        for bond in nodal_cashflows:
            calc_price = price_bond(bond['times'], bond['amounts'], current_spline)
            err = calc_price - bond['market_price']
            errors.append(err)

        return np.array(errors)

# 4. Run the solver!
print("Optimizing zero rates to match market prices...")
result = least_squares(objective_function, initial_guess, ftol=1e-8, xtol=1e-8)

# 5. Extract the optimal zero rates and build our final Calibrated ZCYC Spline!
optimal_zero_rates = result.x
bootstrapped_zcyc = CubicSpline(knot_tenors, optimal_zero_rates, bc_type='natural')

print("Optimization Successful!")
print("-" * 55)
print("Tenor (Yrs) | Market Price | Model Price | Price Error (Rs)")
print("-" * 55)
for i, bond in enumerate(nodal_cashflows):
        final_price = price_bond(bond['times'], bond['amounts'], bootstrapped_zcyc)
        err = final_price - bond['market_price']
        print(f"{bond['tenor']:10.2f}  | {bond['market_price']:12.4f} | {final_price:11.4f} | {err:16.6f}")
print("-" * 55)


#=====================================================================================================
#                    Evaluating our model using official FBIL ZCYC tenors
#====================================================================================================

# 1. Prepare comparison table using the official FBIL ZCYC tenors
    # Filter tenors up to 40 years (matching our bond data limit)
comparison_df = zcyc_df[zcyc_df['Tenor (Year)'] <= 40.0].copy()

    # Evaluate our bootstrapped spline at each official tenor point
comparison_df['Model_ZCYC'] = bootstrapped_zcyc(comparison_df['Tenor (Year)'])
comparison_df['FBIL_ZCYC'] = comparison_df['Zero Coupon%  (Semi-annual)']

    # Compute difference in percentage and in Basis Points (bps)
comparison_df['Diff_%'] = comparison_df['Model_ZCYC'] - comparison_df['FBIL_ZCYC']
comparison_df['Diff_bps'] = comparison_df['Diff_%'] * 100.0

    # 2. Print summary statistical metrics
mae_bps = np.mean(np.abs(comparison_df['Diff_bps']))
max_bps = np.max(np.abs(comparison_df['Diff_bps']))
rmse_bps = np.sqrt(np.mean(comparison_df['Diff_bps']**2))

print("=" * 50)
print("       COMPARISON METRICS (MODEL vs FBIL)       ")
print("=" * 50)
print(f"Mean Absolute Error (MAE): {mae_bps:.2f} bps ({mae_bps/100:.4f}%)")
print(f"Root Mean Square Error:   {rmse_bps:.2f} bps ({rmse_bps/100:.4f}%)")
print(f"Max Difference:           {max_bps:.2f} bps ({max_bps/100:.4f}%)")
print("=" * 50)

    # 3. Create the Comparison Charts
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    # Top Plot: Yield Curves
ax1.plot(comparison_df['Tenor (Year)'], comparison_df['Model_ZCYC'],
             label='Our Bootstrapped ZCYC (Cubic Spline)', color='#1f77b4', linewidth=2.5)
ax1.plot(comparison_df['Tenor (Year)'], comparison_df['FBIL_ZCYC'],
             label='Official FBIL Benchmark ZCYC', color='#d62728', linestyle='--', linewidth=2.0)
ax1.scatter(knot_tenors, optimal_zero_rates, color='black', zorder=5, label='Nodal Knots')

ax1.set_title('Bootstrapped Zero Coupon Yield Curve vs. FBIL Benchmark (18-Sep-2026)', fontsize=13,
  fontweight='bold')
ax1.set_ylabel('Zero Rate (% p.a. Semi-Annual)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True, shadow=True)

    # Bottom Plot: Spread Discrepancy in Basis Points
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
ax2.plot(comparison_df['Tenor (Year)'], comparison_df['Diff_bps'], color='#2ca02c', linewidth=1.8)
ax2.fill_between(comparison_df['Tenor (Year)'], comparison_df['Diff_bps'], 0, color='#2ca02c', alpha=0.15)

ax2.set_xlabel('Maturity Tenor (Years)', fontsize=11)
ax2.set_ylabel('Difference (bps)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('zcyc_comparison_plot.png', dpi=300)
plt.show()
print("Saved comparison chart as 'zcyc_comparison_plot.png'!")


#==================================================================================================
#                               identifying the issues and refining the model
#==================================================================================================
print('''seems like our existing model has overfitted the data
The possible reason might be the following:
\n\nWe have used the clean price in our model which is the real culprit, because in practice when a bond is sold the buyer has to pay the dirty price, which accounts for accured interest as well ''')


print("\nRefining model based on drity price calculation")
# Function to calculate Accrued Interest and Dirty Price
def get_bond_details(coupon, maturity_date, clean_price, val_date=VALUATION_DATE):
        coupon_pmt = coupon / 2.0

        # Generate coupon schedule
        cashflow_dates = []
        curr_date = maturity_date
        while curr_date > val_date:
            cashflow_dates.append(curr_date)
            curr_date = curr_date - pd.DateOffset(months=6)
        cashflow_dates = sorted(cashflow_dates)

        # Previous coupon date was 6 months before the next coupon
        prev_coupon_date = cashflow_dates[0] - pd.DateOffset(months=6)

        # Accrued interest = coupon * (days since last coupon / 365)
        days_accrued = (val_date - prev_coupon_date).days
        accrued_interest = coupon * (days_accrued / 365.0)
        dirty_price = clean_price + accrued_interest

        times = np.array([(d - val_date).days / 365.25 for d in cashflow_dates])
        amounts = np.full(len(cashflow_dates), coupon_pmt)
        amounts[-1] += 100.0

        return times, amounts, dirty_price
# 1. Precompute cash flows and DIRTY PRICES for nodal bonds
nodal_cashflows = []
for idx, row in nodal_bonds.iterrows():
        times, amounts, dirty_price = get_bond_details(row['Coupon'], row['Maturity_dt'], row['Price(Rs)'])
        nodal_cashflows.append({
            'times': times,
            'amounts': amounts,
            'market_price': dirty_price,  # Using the true dirty price!
            'tenor': row['Tenor']
        })

knot_tenors = np.array([bond['tenor'] for bond in nodal_cashflows])
initial_guess = nodal_bonds['YTM% p.a. (Semi-Annual)'].values

# 2. Re-run optimization
def objective_function(guess_rates):
        current_spline = CubicSpline(knot_tenors, guess_rates, bc_type='natural')
        errors = []
        for bond in nodal_cashflows:
            calc_price = price_bond(bond['times'], bond['amounts'], current_spline)
            errors.append(calc_price - bond['market_price'])
        return np.array(errors)

result = least_squares(objective_function, initial_guess, ftol=1e-8, xtol=1e-8)
optimal_zero_rates = result.x
bootstrapped_zcyc = CubicSpline(knot_tenors, optimal_zero_rates, bc_type='natural')
print("Optimization re-run successfully with Accrued Interest accounted for!")

# 1. Prepare comparison table using the official FBIL ZCYC tenors
    # Filter tenors up to 40 years (matching our bond data limit)
comparison_df = zcyc_df[zcyc_df['Tenor (Year)'] <= 40.0].copy()

    # Evaluate our bootstrapped spline at each official tenor point
comparison_df['Model_ZCYC'] = bootstrapped_zcyc(comparison_df['Tenor (Year)'])
comparison_df['FBIL_ZCYC'] = comparison_df['Zero Coupon%  (Semi-annual)']

    # Compute difference in percentage and in Basis Points (bps)
comparison_df['Diff_%'] = comparison_df['Model_ZCYC'] - comparison_df['FBIL_ZCYC']
comparison_df['Diff_bps'] = comparison_df['Diff_%'] * 100.0

    # 2. Print summary statistical metrics
mae_bps = np.mean(np.abs(comparison_df['Diff_bps']))
max_bps = np.max(np.abs(comparison_df['Diff_bps']))
rmse_bps = np.sqrt(np.mean(comparison_df['Diff_bps']**2))

print("=" * 50)
print("       COMPARISON METRICS (MODEL vs FBIL)       ")
print("=" * 50)
print(f"Mean Absolute Error (MAE): {mae_bps:.2f} bps ({mae_bps/100:.4f}%)")
print(f"Root Mean Square Error:   {rmse_bps:.2f} bps ({rmse_bps/100:.4f}%)")
print(f"Max Difference:           {max_bps:.2f} bps ({max_bps/100:.4f}%)")
print("=" * 50)

    # 3. Create the Comparison Charts
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    # Top Plot: Yield Curves
ax1.plot(comparison_df['Tenor (Year)'], comparison_df['Model_ZCYC'],
             label='Our Bootstrapped ZCYC (Cubic Spline)', color='#1f77b4', linewidth=2.5)
ax1.plot(comparison_df['Tenor (Year)'], comparison_df['FBIL_ZCYC'],
             label='Official FBIL Benchmark ZCYC', color='#d62728', linestyle='--', linewidth=2.0)
ax1.scatter(knot_tenors, optimal_zero_rates, color='black', zorder=5, label='Nodal Knots')

ax1.set_title('Bootstrapped Zero Coupon Yield Curve vs. FBIL Benchmark (18-Sep-2026)', fontsize=13,
  fontweight='bold')
ax1.set_ylabel('Zero Rate (% p.a. Semi-Annual)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True, shadow=True)

    # Bottom Plot: Spread Discrepancy in Basis Points
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
ax2.plot(comparison_df['Tenor (Year)'], comparison_df['Diff_bps'], color='#2ca02c', linewidth=1.8)
ax2.fill_between(comparison_df['Tenor (Year)'], comparison_df['Diff_bps'], 0, color='#2ca02c', alpha=0.15)

ax2.set_xlabel('Maturity Tenor (Years)', fontsize=11)
ax2.set_ylabel('Difference (bps)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('zcyc_comparison_plot.png', dpi=300)
plt.show()
print("Saved comparison chart as 'zcyc_comparison_plot_refined.png'!")



#===============================================================================================================
#                                                  Refining the model further 
#===============================================================================================================
print("Hurray ! the model is performing better than before, but two major issue can still be addressed")
print("--------------Problem 1---------------\n")
print("\nWe have used 12 knots on 12 bonds which forces the cubic Spline to bend drastically in order to hit every single bond\'s price 100% exactly")
print("\nThis issue can be resolved using the standard FBIL methods : The smoothing Spline")
print("\nWe look forward to find lesser knots to fit the curve well")
print("\nWe increase nodal bonds from 12 to 25 but keep only 6 knots, this creates a a system of outvoting outliers, in the sense that if certain bond is outlier its neighbouring bonds outvote it and reduce its weightage.")

print("\n--------------Problem 2---------------\n")
print("\nA large tenure bond \(for example, 30 yrs\) is super sensitive to interest rates compared to a short tenure bond \(for example 6 months\) which is barely sensitive")
print("\nTo address this issue we add duration weights so that we give shorter and medium bonds an equal and balanced voice in guiding the curve")


# 1. Filter vanilla, active G-Sec bonds (clean coupon > 5% and realistic YTM)
active_bonds = gsec_df[(gsec_df['Coupon'] >= 5.0) &
                           (gsec_df['YTM% p.a. (Semi-Annual)'] >= 5.0) &
                           (gsec_df['YTM% p.a. (Semi-Annual)'] <= 9.0)].copy()

    # Sort by tenor and pick ~25 evenly distributed bonds across the timeline
active_bonds = active_bonds.sort_values('Tenor').reset_index(drop=True)
step = max(1, len(active_bonds) // 25)
fit_bonds = active_bonds.iloc[::step].copy().reset_index(drop=True)

print(f"Using {len(fit_bonds)} active bonds to calibrate a smooth curve.")

    # 2. Precompute cash flows and dirty prices for all 25 bonds
fit_cashflows = []
for idx, row in fit_bonds.iterrows():
        times, amounts, dirty_price = get_bond_details(row['Coupon'], row['Maturity_dt'], row['Price(Rs)'])
        fit_cashflows.append({
            'times': times,
            'amounts': amounts,
            'market_price': dirty_price,
            'tenor': row['Tenor'],
            'duration_weight': 1.0 / np.sqrt(max(0.5, row['Tenor'])) # Weights short and long bonds fairly
        })

    # 3. Use 6 stable, strategic anchor knots (NOT 12!)
knot_tenors = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 40.0])
initial_guess = np.array([5.75, 6.30, 6.80, 7.05, 7.80, 8.20])

    # 4. Objective function: minimize weighted price errors across all 25 bonds
def smooth_objective(guess_rates):
        spline = CubicSpline(knot_tenors, guess_rates, bc_type='natural')
        residuals = []
        for bond in fit_cashflows:
            calc_price = price_bond(bond['times'], bond['amounts'], spline)
            # Weighted error
            err = (calc_price - bond['market_price']) * bond['duration_weight']
            residuals.append(err)
        return np.array(residuals)

    # 5. Optimize!
res = least_squares(smooth_objective, initial_guess, ftol=1e-9, xtol=1e-9)
optimal_zero_rates = res.x
bootstrapped_zcyc = CubicSpline(knot_tenors, optimal_zero_rates, bc_type='natural')
print("Smooth calibration complete!")


#Prepare comparison table using the official FBIL ZCYC tenors
# Filter tenors up to 40 years (matching our bond data limit)
comparison_df = zcyc_df[zcyc_df['Tenor (Year)'] <= 40.0].copy()

    # Evaluate our bootstrapped spline at each official tenor point
comparison_df['Model_ZCYC'] = bootstrapped_zcyc(comparison_df['Tenor (Year)'])
comparison_df['FBIL_ZCYC'] = comparison_df['Zero Coupon%  (Semi-annual)']

    # Compute difference in percentage and in Basis Points (bps)
comparison_df['Diff_%'] = comparison_df['Model_ZCYC'] - comparison_df['FBIL_ZCYC']
comparison_df['Diff_bps'] = comparison_df['Diff_%'] * 100.0

    # 2. Print summary statistical metrics
mae_bps = np.mean(np.abs(comparison_df['Diff_bps']))
max_bps = np.max(np.abs(comparison_df['Diff_bps']))
rmse_bps = np.sqrt(np.mean(comparison_df['Diff_bps']**2))

print("=" * 50)
print("       COMPARISON METRICS (MODEL vs FBIL)       ")
print("=" * 50)
print(f"Mean Absolute Error (MAE): {mae_bps:.2f} bps ({mae_bps/100:.4f}%)")
print(f"Root Mean Square Error:   {rmse_bps:.2f} bps ({rmse_bps/100:.4f}%)")
print(f"Max Difference:           {max_bps:.2f} bps ({max_bps/100:.4f}%)")
print("=" * 50)

    # 3. Create the Comparison Charts
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    # Top Plot: Yield Curves
ax1.plot(comparison_df['Tenor (Year)'], comparison_df['Model_ZCYC'],
             label='Our Bootstrapped ZCYC (Cubic Spline)', color='#1f77b4', linewidth=2.5)
ax1.plot(comparison_df['Tenor (Year)'], comparison_df['FBIL_ZCYC'],
             label='Official FBIL Benchmark ZCYC', color='#d62728', linestyle='--', linewidth=2.0)
ax1.scatter(knot_tenors, optimal_zero_rates, color='black', zorder=5, label='Nodal Knots')

ax1.set_title('Bootstrapped Zero Coupon Yield Curve vs. FBIL Benchmark (18-Sep-2026)', fontsize=13,
  fontweight='bold')
ax1.set_ylabel('Zero Rate (% p.a. Semi-Annual)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True, shadow=True)

    # Bottom Plot: Spread Discrepancy in Basis Points
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
ax2.plot(comparison_df['Tenor (Year)'], comparison_df['Diff_bps'], color='#2ca02c', linewidth=1.8)
ax2.fill_between(comparison_df['Tenor (Year)'], comparison_df['Diff_bps'], 0, color='#2ca02c', alpha=0.15)

ax2.set_xlabel('Maturity Tenor (Years)', fontsize=11)
ax2.set_ylabel('Difference (bps)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('zcyc_comparison_plot.png', dpi=300)
plt.show()
print("Saved comparison chart as 'zcyc_comparison_plot_re_refined.png'!")


# ==============================================================================
#            BONUS TASK: SDL STRIPS PRICING ENGINE (RBI ANNEX 4)
# ==============================================================================

# 1. Load the SDL Valuation dataset and the SDL ZCYC
sdl_df = pd.read_csv('data/sdl_18Sep2026(SDL).csv', skiprows=4)
sdl_df.columns = sdl_df.columns.str.strip()

sdlzcyc_df = pd.read_csv('data/sdlzcyc_18Sep2026(SDL_ZCYC).csv', skiprows=4)
sdlzcyc_df.columns = sdlzcyc_df.columns.str.strip()

    # Create a spline function for SDL spot rates
sdl_spline = CubicSpline(sdlzcyc_df['Tenor (Year)'], sdlzcyc_df['Zero Coupon%  (Semi-annual)'],
  bc_type='natural')

    # 2. Select a long-dated, high-quality benchmark SDL (e.g. a ~10-year bond maturing in 2036)
sdl_df['Maturity_dt'] = pd.to_datetime(sdl_df['Maturity (dd-mmm-yyyy)'])
sdl_df['Tenor'] = (sdl_df['Maturity_dt'] - VALUATION_DATE).dt.days / 365.25

    # Pick a clean 10-year SDL (Tenor between 9.8 and 10.2 years)
candidate_sdls = sdl_df[(sdl_df['Tenor'] >= 9.8) & (sdl_df['Tenor'] <= 10.2)]
selected_sdl = candidate_sdls.iloc[0]

print("=" * 65)
print("             SELECTED STATE DEVELOPMENT LOAN (SDL)              ")
print("=" * 65)
print(f"ISIN:          {selected_sdl['ISIN']}")
print(f"Description:   {selected_sdl['Description']}")
print(f"Coupon Rate:   {selected_sdl['Coupon']}% per annum")
print(f"Maturity Date: {selected_sdl['Maturity (dd-mmm-yyyy)']} (Tenor: {selected_sdl['Tenor']:.2f} yrs)")
print(f"Market Price:  Rs {selected_sdl['Price(Rs)']:.4f} per Rs 100 face value")
print("=" * 65)

    # 3. Decompose into individual STRIPS (Coupon STRIPS + Principal STRIP)
coupon_amount = selected_sdl['Coupon'] / 2.0  # Semi-annual coupon per Rs 100

    # Generate the 6-month coupon dates
cashflow_dates = []
curr = selected_sdl['Maturity_dt']
while curr > VALUATION_DATE:
        cashflow_dates.append(curr)
        curr = curr - pd.DateOffset(months=6)
cashflow_dates = sorted(cashflow_dates)

strips_records = []
for k, date in enumerate(cashflow_dates, start=1):
        t_years = (date - VALUATION_DATE).days / 365.25
        spot_rate = float(sdl_spline(t_years))

        # Standard Coupon STRIP nomenclature: GS<DD><MON><YYYY>C
        strip_name = f"CS {date.strftime('%d %b %Y').upper()}"

        # RBI Semi-annual Discounting: PV = CashFlow / (1 + rate/200)^k
        raw_pv = coupon_amount / np.power(1.0 + (spot_rate / 200.0), k)

        strips_records.append({
            'Period (k)': k,
            'STRIP Type': 'Coupon STRIP',
            'Nomenclature': strip_name,
            'Maturity Date': date.strftime('%d-%b-%Y'),
            'Tenor (Yrs)': round(t_years, 2),
            'Cash Flow (Rs)': coupon_amount,
            'Spot Rate (%)': round(spot_rate, 4),
            'Raw PV (Rs)': raw_pv
        })

    # Add the final Principal STRIP (PS)
final_date = cashflow_dates[-1]
final_k = len(cashflow_dates)
final_t = (final_date - VALUATION_DATE).days / 365.25
final_spot = float(sdl_spline(final_t))
ps_raw_pv = 100.0 / np.power(1.0 + (final_spot / 200.0), final_k)

strips_records.append({
        'Period (k)': final_k,
        'STRIP Type': 'Principal STRIP',
        'Nomenclature': f"PS {final_date.strftime('%d %b %Y').upper()}",
        'Maturity Date': final_date.strftime('%d-%b-%Y'),
        'Tenor (Yrs)': round(final_t, 2),
        'Cash Flow (Rs)': 100.00,
        'Spot Rate (%)': round(final_spot, 4),
        'Raw PV (Rs)': ps_raw_pv
    })

strips_table = pd.DataFrame(strips_records)

    # 4. Apply RBI Annex 4 Normalization
sum_pv_strips = strips_table['Raw PV (Rs)'].sum()
market_value = selected_sdl['Price(Rs)']
book_value = 100.00  # Par book value assumption
carrying_target = min(book_value, market_value)

normalization_factor = carrying_target / sum_pv_strips
strips_table['Normalized Value (Rs)'] = strips_table['Raw PV (Rs)'] * normalization_factor

print("\n--- RBI ANNEX 4 NORMALIZATION SUMMARY ---")
print(f"Sum Total of PV of all STRIPS:    Rs {sum_pv_strips:.4f}")
print(f"Target Value (min[Book, Mkt]):     Rs {carrying_target:.4f}")
print(f"Normalization Factor (lambda):     {normalization_factor:.6f}")
print(f"Sum of Normalized STRIPS:          Rs {strips_table['Normalized Value (Rs)'].sum():.4f}")
print("=" * 65)

    # Display the first 5 and last 3 STRIPS
print("\nSample of Priced STRIPS Schedule:")
print(strips_table[['STRIP Type', 'Maturity Date', 'Cash Flow (Rs)', 'Spot Rate (%)', 'Raw PV (Rs)', 'Normalized Value (Rs)']].to_string(index=False))



#============================================================================================================
#                            Exporting the results
#============================================================================================================
# 1. Export the priced STRIPS schedule to a clean CSV for submission
strips_table.to_csv('priced_sdl_strips.csv', index=False)
print("Saved complete STRIPS schedule to 'priced_sdl_strips.csv'!")

    # 2. Plot the STRIPS Price Profile
plt.figure(figsize=(11, 5))

    # Plot bars for Coupon STRIPS and Principal STRIP
cs_mask = strips_table['STRIP Type'] == 'Coupon STRIP'
ps_mask = strips_table['STRIP Type'] == 'Principal STRIP'

plt.bar(strips_table[cs_mask]['Tenor (Yrs)'], strips_table[cs_mask]['Normalized Value (Rs)'],
            width=0.35, color='#1f77b4', label='Coupon STRIPS (Semi-Annual Cash Flows)')

plt.bar(strips_table[ps_mask]['Tenor (Yrs)'], strips_table[ps_mask]['Normalized Value (Rs)'],
            width=0.45, color='#ff7f0e', label='Principal STRIP (Rs 100 Face Value)')

plt.title(f"STRIPS Valuation Profile: {selected_sdl['Description']} (RBI Annex 4)", fontsize=12,
  fontweight='bold')
plt.xlabel('Maturity (Years from Today)', fontsize=11)
plt.ylabel('Carrying / Normalized Value (Rs)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, shadow=True)

    # Annotate the values of the first coupon, last coupon, and principal
first_cs = strips_table[cs_mask].iloc[0]
last_cs = strips_table[cs_mask].iloc[-1]
ps = strips_table[ps_mask].iloc[0]

plt.annotate(f"First Coupon: Rs {first_cs['Normalized Value (Rs)']:.2f}",
                 xy=(first_cs['Tenor (Yrs)'], first_cs['Normalized Value (Rs)']),
                 xytext=(first_cs['Tenor (Yrs)']+0.5, first_cs['Normalized Value (Rs)']+3),
                 arrowprops=dict(arrowstyle="->", color="black"))

plt.annotate(f"Principal STRIP: Rs {ps['Normalized Value (Rs)']:.2f}",
                 xy=(ps['Tenor (Yrs)'], ps['Normalized Value (Rs)']),
                 xytext=(ps['Tenor (Yrs)']-3.5, ps['Normalized Value (Rs)']-8),
                 arrowprops=dict(arrowstyle="->", color="black"))

plt.tight_layout()
plt.savefig('sdl_strips_valuation_plot.png', dpi=300)
plt.show()
print("Saved STRIPS chart as 'sdl_strips_valuation_plot.png'!")

