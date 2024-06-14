# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run training script
RUN python -m src.train_pipeline

# Expose the port the app runs on
EXPOSE 8000

# Run the Flask app
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]

