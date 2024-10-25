# Use an official Python runtime as a base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to Cloud Run
EXPOSE 8080

# Set environment variable to specify the port for Cloud Run
ENV PORT 8080

# Run the application using gunicorn (adjust "main:app" as needed)
CMD exec gunicorn --bind :8080 flask_app:app