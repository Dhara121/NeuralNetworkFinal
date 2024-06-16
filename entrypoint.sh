#!/bin/bash

# Train the model
python3 /app/src/train_pipeline.py

# Run predictions
python3 /app/src/predict.py

# Keep the container running (optional, if you need it to stay up for serving predictions)
tail -f /dev/null

