import os
import requests
import zipfile
from IPython import get_ipython

# Define URL and file paths
url = "https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip"
zip_path = "classification_data.zip"
extract_path = "./"  # Current directory

# Download the file
print("Downloading dataset...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(zip_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")
else:
    raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Unzip the file
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction complete.")

# Delete the ZIP file
os.remove(zip_path)
print("ZIP file deleted.")

# Enable autoreload in Jupyter Notebook
ipython = get_ipython()
if ipython is not None:
    print("Enabling autoreload...")
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")