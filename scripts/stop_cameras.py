import requests
import sys

BASE_URL = "http://localhost:8000/api"

def stop_all_cameras():
    try:
        # 1. Get the list of devices
        response = requests.get(f"{BASE_URL}/devices")
        response.raise_for_status()
        devices = response.json()
        
        active_cameras = devices.get("active", [])
        if not active_cameras:
            print("No active cameras to stop.")
            return

        print(f"Stopping {len(active_cameras)} active cameras: {', '.join(active_cameras)}")
        
        # 2. Stop each active camera
        for camera_id in active_cameras:
            stop_url = f"{BASE_URL}/cameras/{camera_id}/stop"
            res = requests.post(stop_url)
            if res.status_code == 200:
                print(f"Successfully stopped {camera_id}")
            else:
                print(f"Failed to stop {camera_id}: {res.status_code} {res.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the ViKi server. Is it running at http://localhost:8000?")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    stop_all_cameras()
