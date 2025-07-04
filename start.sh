#!/bin/sh
set -e

# Common S3 sync functionality for both Airflow and Streamlit
start_s3_sync() {
  echo "Starting S3 sync (if BUCKET_NAME is set)..."
  if [ -n "$BUCKET_NAME" ]; then
    mkdir -p /app/saved_models
    aws s3 sync s3://"$BUCKET_NAME"/saved_models /app/saved_models
    echo "Saved models sync complete."
  else
    echo "BUCKET_NAME is not set. Skipping S3 sync."
  fi
}

# Airflow section
if [ "$1" = "airflow" ]; then
  start_s3_sync  # Perform the S3 sync

  echo "Migrating Airflow DB..."
  airflow db upgrade
  echo "Airflow DB migration completed."

  echo "Checking if Admin user exists..."
  if ! airflow users list | grep -w "$AIRFLOW_USERNAME" > /dev/null 2>&1; then
    echo "Creating Admin user..."
    airflow users create \
      --email "$AIRFLOW_EMAIL" \
      --firstname "Admin" \
      --lastname "User" \
      --password "$AIRFLOW_PASSWORD" \
      --role "Admin" \
      --username "$AIRFLOW_USERNAME"
  else
    echo "Admin user exists."
  fi

  # Start Airflow scheduler in the background
  nohup airflow scheduler &

  # Start Airflow webserver
  airflow webserver

# Flask + Gunicorn section
elif [ "$1" = "app" ]; then
  start_s3_sync  # Perform the S3 sync

  echo "Starting Flask app with Gunicorn..."
  exec gunicorn app:app --bind 0.0.0.0:8501 --workers 4

else
  echo "Unknown service: $1"
  exec "$@"
fi
