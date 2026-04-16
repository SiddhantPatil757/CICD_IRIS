from ML_CICD_IRES.src.train import train
from ML_CICD_IRES.src.predict import predict

def test_model():
    train()
    sample = [5.1,3.5,1.4,0.2]
    result = predict(sample)

    assert result is not None