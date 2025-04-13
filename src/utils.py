import matplotlib.pyplot as plt
import torch

def show_image(tensor_img, title="Image"):
    # Unnormalize if normalized
    if tensor_img.shape[0] == 3:
        tensor_img = tensor_img.permute(1, 2, 0).numpy()
    plt.imshow(tensor_img)
    plt.title(title)
    plt.axis("off")
    plt.show()
