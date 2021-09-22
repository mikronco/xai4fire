#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh
export CUDA_VISIBLE_DEVICES=1

echo "Running experiment optuna lstm_temporal_cls"
python run.py experiment=cnn_spatial_cls ++model.weight_decay=0.02 ++model.hidden_size=16 ++model.lr=0.0001

echo "Running experiment optuna cnn_spatial_cls"
python run.py experiment=lstm_temporal_cls ++model.weight_decay=0.001 ++model.hidden_size=64 ++model.lr=0.005


echo "Running experiment optuna clstm_spatiotemporal_cls"
python run.py experiment=clstm_spatiotemporal_cls ++model.weight_decay=0.03 ++model.hidden_size=16 ++model.lr=0.0001
