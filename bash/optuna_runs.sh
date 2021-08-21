#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

echo "Running experiment optuna cnn_spatial_cls"
python run.py -m hparams_search=greecefire_optuna experiment=cnn_spatial_cls

echo "Running experiment optuna lstm_temporal_cls"
python run.py -m hparams_search=greecefire_optuna experiment=lstm_temporal_cls

echo "Running experiment optuna clstm_spatiotemporal_cls"
python run.py -m hparams_search=greecefire_optuna experiment=clstm_spatiotemporal_cls

