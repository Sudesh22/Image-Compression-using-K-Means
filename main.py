import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin_min
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((400, 200))  # Adjust the size as needed
    image = np.array(image)
    return image

image_path = "cats.jpg"  # Original image path
original_image = load_image(image_path)

original_shape = original_image.shape
reshaped_image = original_image.reshape(-1, 3)

n_colors = 256  # Adjust the number of colors to your preference
kmeans = KMeans(n_clusters=n_colors, random_state=42)
kmeans.fit(reshaped_image)

compressed_image = kmeans.cluster_centers_[kmeans.labels_]

compressed_image = compressed_image.reshape(original_shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(compressed_image.astype(np.uint8))
axes[1].set_title("Compressed Image")
axes[1].axis("off")
plt.show()

compressed_image_path = "compressed_image.jpg"  # Replace with your desired path
compressed_image = Image.fromarray(compressed_image.astype(np.uint8))
compressed_image.save(compressed_image_path)
