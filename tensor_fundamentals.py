import numpy as np

print("=" * 65)
print("TENSOR FUNDAMENTALS: 0D TO 4D TENSORS IN NUMPY")
print("=" * 65)

# 1. 0D Tensor (Scalar)
tensor_0d = np.array(42)
print(f"0D Tensor (Scalar):  Value={tensor_0d}, ndim={tensor_0d.ndim}, shape={tensor_0d.shape}, size={tensor_0d.size}")

# 2. 1D Tensor (Vector)
tensor_1d = np.array([7.5, 105.0, 1.0])
print(f"1D Tensor (Vector):  ndim={tensor_1d.ndim}, shape={tensor_1d.shape}, size={tensor_1d.size}")

# 3. 2D Tensor (Matrix - Tabular dataset with 3 samples, 2 features)
tensor_2d = np.array([
    [8.5, 115],
    [7.8, 110],
    [5.5, 90]
])
print(f"2D Tensor (Matrix):  ndim={tensor_2d.ndim}, shape={tensor_2d.shape}, size={tensor_2d.size}")

# 4. 3D Tensor (Time Series / Stock: 2 stocks, 5 trading days, [Open, Close])
tensor_3d = np.random.uniform(100, 200, size=(2, 5, 2))
print(f"3D Tensor (Time):    ndim={tensor_3d.ndim}, shape={tensor_3d.shape}, size={tensor_3d.size}")

# 5. 4D Tensor (Vision Batch: 8 images, 64x64 pixels, 3 RGB channels)
tensor_4d = np.zeros((8, 64, 64, 3), dtype=np.float32)
memory_bytes = tensor_4d.nbytes
print(f"4D Tensor (Vision):  ndim={tensor_4d.ndim}, shape={tensor_4d.shape}, size={tensor_4d.size}")
print(f"   Memory Allocated: {memory_bytes / 1024:.2f} KB")

# 6. Demonstrating the 5D Video Memory footprint calculation
batch_videos, frames, height, width, channels = 4, 1800, 480, 720, 3
total_elements = batch_videos * frames * height * width * channels
raw_gb = (total_elements * 4) / (1024 ** 3)  # 4 bytes per float32
print("-" * 65)
print(f"5D Video Batch Shape: ({batch_videos}, {frames}, {height}, {width}, {channels})")
print(f"Total Elements:       {total_elements:,}")
print(f"Raw RAM Required:     {raw_gb:.2f} GB (Why raw video ML needs compression/sampling)")
print("=" * 65)