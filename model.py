import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# --- Step 1: Load the Dataset ---
try:
    df = pd.read_csv('generated_shielding_data_full.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'generated_shielding_data_full.csv' not found. Please run the data generation script first.")
    exit()

# --- Step 2: Define the Features (X) and Targets (y) ---
# The inputs to the model
features = [
    'Frequency_GHz',
    'sigma_eff_S_m',
    'sigma_star_S_m',
    'SE_R_dB',
    'SE_A_dB',
    'SE_T_dB'
]

# The outputs the model needs to predict
targets = [
    'Biochar_wt',
    'PANI_wt',
    'MWCNT_wt'
]

X = df[features]
y = df[targets]

print(f"✅ Features (Inputs):\n{features}")
print(f"✅ Targets (Outputs):\n{targets}")

# --- Step 3: Split the Data into Training and Testing Sets ---
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- Step 4: Initialize and Train the Random Forest Model ---
# n_estimators is the number of trees in the forest. More is often better but takes longer.
# random_state ensures we get the same result every time we run the code.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available CPU cores

print("\nTraining the Random Forest model... 🌳")
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- Step 5: Evaluate the Model on the Test Data ---
print("\nEvaluating model performance on the test set...")
y_pred = model.predict(X_test)

# Calculate metrics for each target variable
mae_biochar = mean_absolute_error(y_test['Biochar_wt'], y_pred[:, 0])
mae_pani = mean_absolute_error(y_test['PANI_wt'], y_pred[:, 1])
mae_mwcnt = mean_absolute_error(y_test['MWCNT_wt'], y_pred[:, 2])

r2_biochar = r2_score(y_test['Biochar_wt'], y_pred[:, 0])
r2_pani = r2_score(y_test['PANI_wt'], y_pred[:, 1])
r2_mwcnt = r2_score(y_test['MWCNT_wt'], y_pred[:, 2])

print("\n--- Performance Metrics ---")
print(f"Mean Absolute Error (MAE):")
print(f"  - Biochar (wt%): {mae_biochar:.3f}")
print(f"  - PANI (wt%):    {mae_pani:.3f}")
print(f"  - MWCNT (wt%):   {mae_mwcnt:.3f}")
print("\n(MAE tells you the average error in the model's prediction for each component.)")


print(f"\nR-squared (R²) Score:")
print(f"  - Biochar: {r2_biochar:.3f}")
print(f"  - PANI:    {r2_pani:.3f}")
print(f"  - MWCNT:   {r2_mwcnt:.3f}")
print("\n(R² Score of 1.0 is a perfect prediction. Values close to 1.0 are excellent.)")

# --- Step 6: Use the Trained Model for a New Prediction ---
# Let's say you want to design a material with the following properties:
target_properties = {
    'Frequency_GHz': 10.0,
    'sigma_eff_S_m': 12.86,
    'sigma_star_S_m': 1.286,
    'SE_R_dB': 18.17,
    'SE_A_dB': 5.53,
    'SE_T_dB': 23.7
}

# Convert the dictionary to a NumPy array in the correct feature order
input_data = np.array([target_properties[feat] for feat in features]).reshape(1, -1)

# Predict the composition
predicted_composition = model.predict(input_data)

print("\n--- 🧪 Example Prediction ---")
print(f"For desired properties:\n{target_properties}")
print("\nPredicted material composition:")
print(f"  - Biochar: {predicted_composition[0][0]:.2f} wt%")
print(f"  - PANI:    {predicted_composition[0][1]:.2f} wt%")
print(f"  - MWCNT:   {predicted_composition[0][2]:.2f} wt%")