import numpy as np


def main():
    X = np.array(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    )
    
    w = np.array([0.1, 0.2, 0.3])
    b = 0.5
    
    scores = X @ w + b
    
    print("X:")
    print(X)
    print("X shape:", X.shape)

    print("w:", w)
    print("w shape:", w.shape)

    print("scores:", scores)
    print("scores shape:", scores.shape)


if __name__ == "__main__":
    main()