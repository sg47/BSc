## Requirements

* `Tensorflow 1.8` (recomended) 
* `python packages` such as opencv, matplotlib

## Run 3net on webcam stream

To run 3net, just launch

```
sh get_checkpoint.sh
python webcam.py --checkpoint_dir /checkpoint/3DV18/3net --mode [0,1,2]
```

While the demo is running, you can press:

* 'm' to change mode (0: depth-from-mono, 1: depth + view synthesis, 2: depth + view + SGM)
* 'p' to pause the stream
* 'ESC' to quit

## Train 3net from scratch

Code for training will be (eventually) uploaded.
Meanwhile, you can train 3net by embedding it into https://github.com/mrharicot/monodepth

