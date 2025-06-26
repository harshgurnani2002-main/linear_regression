import pickle

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

try:
    print("Scaler feature names:")
    print(scaler.feature_names_in_)
except AttributeError:
    print("❌ Scaler does not store feature names.")
