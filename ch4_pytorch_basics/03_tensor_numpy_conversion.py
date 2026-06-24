import numpy as np
import torch


def main():
    np_array = np.array([1, 2, 3], dtype=np.float32)

    tensor_from_numpy = torch.from_numpy(np_array)

    print("NumPy array:", np_array)
    print("Tensor from NumPy:", tensor_from_numpy)
    print("Tensor dtype:", tensor_from_numpy.dtype)

    np_array[0] = 100

    print()
    print("After modifying NumPy array:")
    print("NumPy array:", np_array)
    print("Tensor from NumPy:", tensor_from_numpy)

    tensor = torch.tensor([4, 5, 6], dtype=torch.float32)

    np_from_tensor = tensor.numpy()

    print()
    print("Tensor:", tensor)
    print("NumPy from Tensor:", np_from_tensor)

    tensor[0] = 400

    print()
    print("After modifying Tensor:")
    print("Tensor:", tensor)
    print("NumPy from Tensor:", np_from_tensor)


if __name__ == "__main__":
    main()
