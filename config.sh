#!/bin/bash
poetry add torch numpy pandas scikit-learn optuna neptune-client<1.0
poetry add black lint pytest --group dev
