# tak

## training

Training done in python3 (pytorch)
To train your own network, simply have torch, numpy and tqdm installed.
After cloning the repository, running `main.py` in alpha-tak-train will train a model.
You can change the model parameters `res_blocks` (number of residual blocks)
and `filters` (channels in the convolution layers) in the same file if you want a
smaller/larger model.
```
net = TakNetwork(stack_limit=10, res_blocks=10, filters=128)
```
I will work on more convenient operating with a config file soon.

If you want to train on your own data, you can train on ptn files. Only game size 6
supported. The ptn parser requires game result in the ptn headers.
Also, it gets picky about capitalization:
- Stone type (S, C) should be capitalized
- square (a,b,c,d,e,f) should not be capitalized
