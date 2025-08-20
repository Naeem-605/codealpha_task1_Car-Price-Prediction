# predict.py
import joblib
import pandas as pd

# ======================
# Load trained pipeline
# ======================
PIPELINE_PATH = r"E:\Code alpha Internship\Car price prediction\car_price_pipeline.pkl"
pipeline = joblib.load(PIPELINE_PATH)

print("✅ Model loaded successfully!\n")

# ======================
# Take input from user
# ======================
year = int(input("Enter Car Year (e.g., 2018): "))
present_price = float(input("Enter Present Price in lakhs (e.g., 5.59): "))
driven_kms = int(input("Enter Driven Kms (e.g., 27000): "))
fuel_type = input("Enter Fuel Type (Petrol / Diesel / CNG): ")
selling_type = input("Enter Selling Type (Dealer / Individual): ")
transmission = input("Enter Transmission (Manual / Automatic): ")
owner = int(input("Enter Number of Owners before (0/1/3+): "))

# ======================
# Prepare data
# ======================
new_car = {
    "Year": year,
    "Present_Price": present_price,
    "Driven_kms": driven_kms,
    "Fuel_Type": fuel_type,
    "Selling_type": selling_type,
    "Transmission": transmission,
    "Owner": owner
}

new_data = pd.DataFrame([new_car])

# ======================
# Make prediction
# ======================
predicted_price = pipeline.predict(new_data)[0]

print("\n🚗 Predicted Selling Price: {:.2f} lakhs".format(predicted_price))
