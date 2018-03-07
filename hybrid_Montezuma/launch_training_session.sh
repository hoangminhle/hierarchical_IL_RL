#!/bin/bash

if python -c "import tensorflow" &> /dev/null; then
    echo 'all good'
else
    echo 'package not yet installed'
    pip install --upgrade tensorflow-gpu==1.3.0
fi

if python -c "import keras" &> /dev/null; then
    echo 'all good'
else
    echo 'package not yet installed'
    pip install keras==2.1.2
    pip install h5py
fi

python run_hybrid_atari_experiment.py