# Modern RNNS

Codebase for the development of scalable RNNs like DeltaNet and similar.

## How to run
Under projects, find the model you want to run. Each model has a choice of configurations that are in a "configs.py" file in the same directory. The configuration name must be passed as the first argument, and then you can specify additional arguments.

Example Commands:
```
# See the list of valid configurations
python projects/lstm/mainlstm.py --help

# See additional configuration options
python projects/lstm/mainlstm.py [config_name] --help

# Run the bit_parity dataset
PYTHONPATH=. torchrun --standalone projects/lstm/mainlstm.py default --dataset bit_parity

# Play with different sequence lengths:
PYTHONPATH=. torchrun --standalone projects/lstm/mainlstm.py default --dataset bit_parity --max_seq_len 128 --train_seq_len 16,16 --eval_seq_len 64,128
```

## LSTM Details

Currently it's running the 'basic_lstm_model'. To run the Quasi LSTM, change the import in mainlstm.py to:
```
# Remove this line
# from basic_lstm_model import Model
# Add this line
from modellstm import Model
```
