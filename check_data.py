

import joblib

print("="*60)
print("CHECKING DATA STRUCTURE")
print("="*60)

# Load the data file
data_path = 'data/processed_diabetes_data.pkl' 
data = joblib.load(data_path)

print("TYPE of loaded data:", type(data))
print("\nKEYS in data (if it's a dictionary):")
if isinstance(data, dict):
    for key in data.keys():
        print(f"  - {key}")
        if hasattr(data[key], 'shape'):
            print(f"    Shape: {data[key].shape}")
            print(f"    Type: {type(data[key])}")
elif isinstance(data, tuple):
    print(f"\nData is a TUPLE with {len(data)} elements:")
    for i, item in enumerate(data):
        print(f"  Element {i}: Type={type(item)}")
        if hasattr(item, 'shape'):
            print(f"    Shape: {item.shape}")
elif isinstance(data, list):
    print(f"\nData is a LIST with {len(data)} elements")
else:
    print("\nData structure:", data)

print("\n" + "="*60)
print("TRYING TO LOAD MODELS TOO")
print("="*60)
try:
    model = joblib.load('models/logistic_regression_model.pkl')
    print(f"✅ Logistic Regression model loaded")
    print(f"   Model type: {type(model)}")
    if hasattr(model, 'get_params'):
        print(f"   Model class: {model.__class__.__name__}")
except Exception as e:
    print(f" Error loading model: {e}")
