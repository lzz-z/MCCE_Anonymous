import os

# AQE logS endpoint API key
if api_key := os.getenv("AZURE_LOG_S_ML_ENDPOINT_API_KEY"):
    AZURE_LOG_S_ML_ENDPOINT_API_KEY = api_key
else:
    raise RuntimeError("Please set AZURE_LOG_S_ML_ENDPOINT_API_KEY")

if api_key := os.getenv("AZURE_RED_POT_ML_ENDPOINT_API_KEY"):
    AZURE_RED_POT_ML_ENDPOINT_API_KEY = api_key
else:
    raise RuntimeError("Please set AZURE_RED_POT_ML_ENDPOINT_API_KEY")
