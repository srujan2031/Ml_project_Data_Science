# ✅ Use a slim Python base image
FROM python:3.11-slim

# ✅ Set working directory inside container
WORKDIR /app

# ✅ Copy only requirements.txt first (for layer caching)
COPY requirements.txt .

# ✅ Install Python dependencies (including Gunicorn)
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# ✅ Copy all project files to container
COPY . .

# ✅ Expose Flask port
EXPOSE 5000

# ✅ Start the app with Gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "application:app"]
