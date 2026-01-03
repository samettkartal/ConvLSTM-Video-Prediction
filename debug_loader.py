
from generate_presentation_assets import robust_load_model
import traceback

print("Starting debug load...")
try:
    model = robust_load_model()
    if model:
        print("Model loaded successfully.")
    else:
        print("Model load returned None.")
except Exception:
    traceback.print_exc()
