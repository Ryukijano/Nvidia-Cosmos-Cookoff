import requests
import os

# Create directory for sample videos
os.makedirs("./nexar_data/sample_videos", exist_ok=True)

# List of Vecteezy free dashcam video URLs
video_urls = [
    "https://static.vecteezy.com/system/resources/previews/000/001/000/original/dash-cam-footage-of-driving-on-highway.mp4",
    "https://static.vecteezy.com/system/resources/previews/000/001/001/original/dash-cam-footage-of-driving-on-city-street.mp4",
    "https://static.vecteezy.com/system/resources/previews/000/001/003/original/dash-cam-footage-of-city-driving.mp4",
    "https://static.vecteezy.com/system/resources/previews/000/001/004/original/dash-cam-footage-of-night-driving.mp4",
    "https://static.vecteezy.com/system/resources/previews/000/001/005/original/dash-cam-footage-of-driving.mp4",
    "https://static.vecteezy.com/system/resources/previews/000/001/006/original/dash-cam-footage.mp4",
    "https://static.vecteezy.com/system/resources/previews/000/001/007/original/dash-cam-footage.mp4",
    "https://static.vecteezy.com/system/resources/previews/000/001/008/original/dash-cam-footage.mp4",
]

for i, url in enumerate(video_urls):
    filename = f"sample_dashcam_{i+1}.mp4"
    filepath = os.path.join("./nexar_data/sample_videos", filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}, status code: {response.status_code}")
    else:
        print(f"Already exists: {filename}")

print("Sample videos download completed.")
