import sys
sys.path.append("build/Release/")
import mytorch
import struct
import gzip
import numpy as np

def load_mnist_images(file_path):
    if file_path.endswith('.gz'):
        f = gzip.open(file_path, 'rb')
    else:
        f = open(file_path, 'rb')
    
    magic, num_images, rows, cols = struct.unpack('>4I', f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(num_images, rows, cols)
    f.close()
    return images

def load_mnist_labels(file_path):
    if file_path.endswith('.gz'):
        f = gzip.open(file_path, 'rb')
    else:
        f = open(file_path, 'rb')
    magic, num_labels = struct.unpack('>2I', f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()
    return labels


train_images_path = './mnist_data/train-images-idx3-ubyte.gz'
train_labels_path = './mnist_data/train-labels-idx1-ubyte.gz'
test_images_path = './mnist_data/t10k-images-idx3-ubyte.gz'
test_labels_path = './mnist_data/t10k-labels-idx1-ubyte.gz'
train_images = mytorch.tensor_from_numpy(load_mnist_images(train_images_path))
train_labels = mytorch.tensor_from_numpy(load_mnist_labels(train_labels_path))
test_images = mytorch.tensor_from_numpy(load_mnist_images(test_images_path))
test_labels = mytorch.tensor_from_numpy(load_mnist_labels(test_labels_path))
#test the shape
print(train_images.shape())
print(train_labels.shape())
print(test_images.shape())
print(test_labels.shape())
