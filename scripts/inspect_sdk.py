import viki_sdk
from viki.capture.kinect import KinectBackend

def inspect():
    print("KinectDevice:", dir(viki_sdk.KinectDevice))
    print("KinectCalibration:", dir(viki_sdk.KinectCalibration))
    print("KinectTransformation:", dir(viki_sdk.KinectTransformation))

if __name__ == "__main__":
    inspect()
