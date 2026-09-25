#!/bin/bash
set -e

python src/data_preparation.py
python src/model_training.py
