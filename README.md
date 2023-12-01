# CMO-NRT

* Usage
* python pipeline.py --window 6 --patch 18 --model cnn --disable_tqdm
* window: the length of lookback months
* patch: the size of small patch, should be divided by 180
* model: cnn, rnn, cnn_rnn, currently cnn also captures the temporal correlation in a simple way, and demonstrates the most robust performance
* --disable_tqdm: whether to show the progress bar
