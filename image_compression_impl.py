import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

# Function to perform KMeans clustering for image quantization
def image_compression(image_np, n_colors):
    # 将图像重塑为 (n_samples, 3) 的二维数组，其中每行代表一个像素的 RGB 值
    pixels = image_np.reshape(-1, 3)

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    # 获取聚类中心颜色
    new_colors = kmeans.cluster_centers_.astype(int)
    # 获取每个像素所属的聚类
    labels = kmeans.labels_

    # 用聚类中心的颜色替换图像中的颜色
    compressed_pixels = new_colors[labels].reshape(image_np.shape)
    
    # 将数据类型转换为 uint8
    compressed_pixels = compressed_pixels.astype('uint8')
    
    return compressed_pixels

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)

def __main__():
    # Load and process the image
    image_path = 'favorite_image.png'  
    output_path = 'compressed_image.png'  
    image_np = load_image(image_path)

    # Perform image quantization using KMeans
    n_colors = 8  # Number of colors to reduce the image to, you may change this to experiment
    quantized_image_np = image_compression(image_np, n_colors)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)

if __name__ == '__main__':
    __main__()