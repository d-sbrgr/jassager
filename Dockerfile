# Use an official Python runtime as a base image
FROM python:3.12-slim

# Copy the current directory contents into the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to Cloud Run
EXPOSE 8080

# Set environment variable to specify the port for Cloud Run
ENV PORT 8080

ENV FLASK_APP=flask_app.py
ENV FLASK_ENV=development

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]