# ------------------------------
# Dockerfile for Breast Cancer Flask App
# ------------------------------

# Use full Python image to get pre-built wheels for scikit-learn, numpy, pandas
FROM python:3.12

# Set working directory inside container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install required Python libraries from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Optional: if you want to ensure all common ML libs are present
RUN pip install --no-cache-dir \
    flask \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    scipy \
    joblib

# Copy the rest of your app files
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
