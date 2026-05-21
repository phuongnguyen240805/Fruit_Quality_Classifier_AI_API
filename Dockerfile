# Sử dụng ảnh TensorFlow pre-built làm base image để không cần tải/biên dịch TensorFlow
FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Copy requirements trước để tận dụng Docker cache layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Expose port Flask
EXPOSE 5000

# Chạy Flask bằng python trực tiếp
CMD ["python", "app.py"]
