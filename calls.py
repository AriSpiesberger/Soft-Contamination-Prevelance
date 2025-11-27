import requests
import os

# 1. Get your key
# Using the key you provided for this test.
API_KEY = "Q6ZFAve3yDBpOKdrbzVEi0z5oiExBdwf" 

# 2. Define the endpoint and parameters
API_BASE_URL = "https://api.harmonic.ai/companies"

# Parameters are still passed as 'params'
# This will be added to the URL (e.g., ?website_domain=harmonic.ai)
params = {
    "website_domain": "harmonic.ai"
}

# 3. Construct the CORRECT headers (based on your docs)
headers = {
    "apikey": API_KEY,               # <-- CHANGED THIS LINE
    "accept": "application/json"
}

# --- Make the API Call ---
print(f"Querying {API_BASE_URL} with POST...")

try:
    # Use requests.post(), passing both headers and params
    # This matches your documentation exactly
    response = requests.post(API_BASE_URL, headers=headers, params=params) # <-- CHANGED THIS LINE
    
    response.raise_for_status() # Check for 4xx/5xx errors

    data = response.json()
    print("\n--- Success! API Response ---")
    print(data)

except requests.exceptions.HTTPError as http_err:
    print(f"\nHTTP error occurred: {http_err}")
    print(f"Response content: {response.text}")
except Exception as err:
    print(f"\nAn error occurred: {err}")