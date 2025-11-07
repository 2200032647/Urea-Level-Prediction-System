import numpy as np
import pandas as pd

def generate_synthetic_data(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    soil_ph = rng.uniform(4.5, 8.5, n)  # acidic to alkaline
    soil_moisture = rng.uniform(5, 40, n)  # percent
    temp_c = rng.uniform(5, 40, n)
    rainfall_mm = rng.uniform(0, 200, n)
    nitrogen = rng.uniform(0, 80, n)  # mg/kg
    phosphorus = rng.uniform(0, 50, n)
    potassium = rng.uniform(50, 400, n)
    organic_matter = rng.uniform(0.5, 8.0, n)  # %
    previous_urea = rng.uniform(0, 120, n)  # kg/ha previously applied
    # crop type: 0=rice,1=maize,2=wheat,3=cotton
    crop_type = rng.randint(0, 4, n)

    # Base urea recommendation formula (synthetic):
    # higher nitrogen -> lower recommended urea; poor organic matter -> higher urea; crop type influences need
    base = (120 - nitrogen) * 0.35 + (7 - organic_matter) * 8 + (35 - soil_ph) * 1.5
    crop_factor = np.array([1.1, 0.9, 1.0, 1.2])[crop_type]
    moisture_factor = np.clip((25 - soil_moisture) * 0.3, -10, 20)
    temp_factor = np.clip((25 - temp_c) * 0.2, -8, 10)
    prev_factor = -0.25 * previous_urea

    urea_level = base * crop_factor + moisture_factor + temp_factor + prev_factor
    urea_level = np.clip(urea_level + rng.normal(0, 8, n), 0, 250)  # kg/ha

    df = pd.DataFrame({
        "soil_ph": soil_ph,
        "soil_moisture": soil_moisture,
        "temp_c": temp_c,
        "rainfall_mm": rainfall_mm,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "organic_matter": organic_matter,
        "previous_urea": previous_urea,
        "crop_type": crop_type,
        "urea_level": urea_level
    })
    return df

if __name__ == '__main__':
    df = generate_synthetic_data(1000)
    df.to_csv("data/sample_data.csv", index=False)
    print("Synthetic data saved to data/sample_data.csv")
