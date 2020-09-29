import numpy as np

def mse_loss(gt, pred):
    loss = np.mean((gt - pred) ** 2)
    return loss

if __name__ == '__main__':
    gt = np.array([1, 0, 0, 1])
    pred = np.array([0, 0, 0, 0])
    print(mse_loss(gt, pred))
