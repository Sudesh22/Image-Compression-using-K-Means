# Image-Compression-using-K-Means

This project implements image compression using the K-means clustering algorithm. By applying K-means clustering to the pixel values of an image, similar colors are grouped together. The algorithm then replaces these groups with representative colors, reducing the number of colors used in the image and thereby compressing it. The compressed image can be saved and displayed, allowing for significant reduction in file size while maintaining visual information.

## How it all works
We import the necessary libraries required for this project
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin_min
from PIL import Image
```

To start off we load and preprocess the image before doing any operations on the image. The `load_image()` function takes the image, resizes it and converts it into a numpy array.
```python
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((400, 200)) 
    image = np.array(image)
    return image

image_path = "cats.jpg"  
original_image = load_image(image_path)
```

After that we reshape the image and convert it to a 2D array
```python
original_shape = original_image.shape
reshaped_image = original_image.reshape(-1, 3)
```

The main work of applying K-means clustering is done as follows. We define the clusters of colours as the `n_colours()` and initialise a `kmeans` object of the `Kmeans()` class and then fit the data (our image array) onto it. 
```python
n_colors = 256 
kmeans = KMeans(n_clusters=n_colors, random_state=42)
kmeans.fit(reshaped_image)
```

To carry out the compression we replace the pixel values with cluster centers
```python
compressed_image = kmeans.cluster_centers_[kmeans.labels_]
```

And we reshape the compressed image back to its original shape
```python
compressed_image = compressed_image.reshape(original_shape)
```

We also display the original and compressed images side by side using matplotlib.

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(compressed_image.astype(np.uint8))
axes[1].set_title("Compressed Image")
axes[1].axis("off")
plt.show()
```

We also save the compressed image in the same directory.
```python
compressed_image_path = "compressed_image.jpg"  # Replace with your desired path
compressed_image = Image.fromarray(compressed_image.astype(np.uint8))
compressed_image.save(compressed_image_path)
```