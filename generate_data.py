import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import itertools

# --- Step 1: Define the original experimental data from TABLE I ---
# This data is our ground truth, now including conductivity.
data = {
    'MWCNT_wt': [5, 10, 15, 20, 25, 30, 35, 40],
    'sigma_eff_S_m': [2.44, 5.73, 12.86, 28.70, 63.81, 141.7, 314.8, 698.1],
    'sigma_star_S_m': [0.244, 0.573, 1.286, 2.87, 6.381, 14.176, 31.686, 69.81],
    'SE_R_dB': [12.48, 15.57, 18.17, 20.46, 22.59, 24.59, 26.50, 28.32],
    'SE_A_dB': [3.26, 4.38, 5.53, 6.78, 8.13, 9.64, 11.33, 13.18]
}
df_exp = pd.DataFrame(data)

# --- Step 2: Model the relationships based on MWCNT content ---
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(df_exp[['MWCNT_wt']])

# Model for SE_R (Reflection)
model_ser = LinearRegression()
model_ser.fit(X_poly, df_exp['SE_R_dB'])

# Model for SE_A (Absorption)
model_sea = LinearRegression()
model_sea.fit(X_poly, df_exp['SE_A_dB'])

# **NEW**: Model for effective conductivity (sigma_eff)
# Using a higher degree polynomial to better capture the sharp, non-linear increase
poly_cond = PolynomialFeatures(degree=4, include_bias=False)
X_poly_cond = poly_cond.fit_transform(df_exp[['MWCNT_wt']])
model_cond = LinearRegression()
model_cond.fit(X_poly_cond, df_exp['sigma_eff_S_m'])


# --- Step 3: Model the effect of Frequency on SE ---
def get_frequency_effect(mwcnt_wt, base_se_t, frequency_ghz):
    base_frequency = 10
    slope = (0.01075 * mwcnt_wt)
    return base_se_t + slope * (frequency_ghz - base_frequency)

# --- Step 4: Generate the new, larger dataset ---

# Define generation parameters
mwcnt_range = range(5, 41, 5)
pani_range = range(20, 41, 10)
biochar_range = range(20, 76, 5)
freq_range = np.linspace(8, 12, 21)
noise_std_dev_se = 0.25 # Noise for SE values in dB
noise_std_dev_cond = 0.05 # Relative noise for conductivity (5%)

generated_data = []

# Find all composition combinations that sum to 100
all_combinations = list(itertools.product(mwcnt_range, pani_range, biochar_range))
valid_compositions = [combo for combo in all_combinations if sum(combo) == 100]
print(f"Found {len(valid_compositions)} valid compositions. Generating data...")

for (mwcnt, pani, biochar) in valid_compositions:
    for freq in freq_range:
        # Predict baseline SE values
        mwcnt_poly_features_se = poly.transform(np.array([[mwcnt]]))
        base_ser = model_ser.predict(mwcnt_poly_features_se)[0]
        base_sea = model_sea.predict(mwcnt_poly_features_se)[0]
        base_set = base_ser + base_sea
        
        # **NEW**: Predict baseline conductivity values
        mwcnt_poly_features_cond = poly_cond.transform(np.array([[mwcnt]]))
        pred_sigma_eff = model_cond.predict(mwcnt_poly_features_cond)[0]
        pred_sigma_star = pred_sigma_eff * 0.1 # Based on observed relationship
        
        # Add relative noise to conductivity
        cond_eff_noise = np.random.normal(0, pred_sigma_eff * noise_std_dev_cond)
        cond_star_noise = np.random.normal(0, pred_sigma_star * noise_std_dev_cond)

        # Calculate final SE values including frequency effect
        final_set = get_frequency_effect(mwcnt, base_set, freq)
        scaling_factor = final_set / base_set if base_set > 0 else 1
        final_ser = base_ser * scaling_factor
        final_sea = base_sea * scaling_factor
        
        # Add noise to SE values
        ser_noise = np.random.normal(0, noise_std_dev_se)
        sea_noise = np.random.normal(0, noise_std_dev_se)
        
        generated_data.append({
            'Biochar_wt': biochar,
            'PANI_wt': pani,
            'MWCNT_wt': mwcnt,
            'Frequency_GHz': freq,
            'sigma_eff_S_m': abs(pred_sigma_eff + cond_eff_noise), # use abs() to prevent negative values
            'sigma_star_S_m': abs(pred_sigma_star + cond_star_noise),
            'SE_R_dB': final_ser + ser_noise,
            'SE_A_dB': final_sea + sea_noise,
            'SE_T_dB': (final_ser + ser_noise) + (final_sea + sea_noise)
        })

df_generated = pd.DataFrame(generated_data)

# --- Save and display the results ---
output_filename = 'generated_shielding_data_full.csv'
df_generated.to_csv(output_filename, index=False)

print(f"\nSuccessfully generated {len(df_generated)} data points.")
print(f"Data saved to '{output_filename}'")
print("\nFirst 5 rows of the generated dataset:")
print(df_generated.head())
print("\nLast 5 rows of the generated dataset:")
print(df_generated.tail())