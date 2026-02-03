import os
import requests
import zipfile

def download_dataset():
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    data_dir = "data"
    zip_path = os.path.join(data_dir, "CMAPSSData.zip")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete. Data is ready in the 'data/' folder.")

if __name__ == "__main__":
    download_dataset()
