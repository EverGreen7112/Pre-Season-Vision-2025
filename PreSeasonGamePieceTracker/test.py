import pickle

def main():
    with open('CameraCalibration-main/cameraMatrix.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data)
if __name__ == '__main__':
    main()