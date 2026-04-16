from src.train import train
from src.predict import predict

if __name__ == "__main__":
    train()

    sample = [5.1,3.5,1.4,0.2]
    result = predict(sample)

    print("prediction:",result)