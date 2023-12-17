# RawNet2
Project on making the AntiSpoofing pipeline. This rep contains my implementation of RawNet2 and all the needed metrics

## How to install?

Make sure to follow this guide
```
git clone https://github.com/aizamaksutova/RawNet2.git
cd RawNet2
pip install -r requirements.txt
```

## How to inference?

First, you should create a directory (e.g. inference/) with all the wavs you want to rate as spoof or bona-fide.

Then perform these:

```
python3 test.py -m checkpoint.pth -inf inference/
```
-inf inference is a directory where your samples are stored.


For inferencing my model exactly you should do these steps:
```
python3 data.py
python3 test.py -m model_rawnet.pth -inf inference/
```
You can look at the results in your output logs

## How to train the model by yourself?
In order to train the model you would need to perform simple steps, but wait for a long time for them to actually download all the data which is a ASV Dataset for antispoofing model training

```
#you need to have a kaggle.json file from your kaggle account
chmod a+x prepare_data.sh
./prepare_data.sh
python3 train.py -c config.json
```
All the other parameters are manually stored in the config.json, but you can look up the config options in train.py in order to change everything right from terminal.


## Wandb report

Here is the [link](https://wandb.ai/aamaksutova/RawNet2/reports/RawNet2--Vmlldzo2Mjg0Njg4?accessToken=qylmiorkilpmmqqrbq1kbqx3gd6ewpx88d6870pr63myxzxdfhokrocak47fnljp) to my wandb report with all the architecture explanation and wavs with their scores on being spoofed or bona-fide

