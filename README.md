# HiFiGAN
Project on GAN-based model capable of generating high fidelity speech. This rep contains my implementation of HiFiGAN model and all the steps to reimplement the pipeline

## How to install?

Make sure to follow this guide
```
git clone https://github.com/aizamaksutova/Vocoder.git
cd Vocoder
pip install -r requirements.txt
```

## How to inference?

First, you should create a file (e.g. samples.wav) with all the phrases you want to reproduce.

Then perform these:

```
python3 test.py -c config.json -r checkpoint.pth -t inference_data -o output --device='gpu'
```
-t inference_data is a directory where your samples are stored and output is a directory where the output results will be stored. 
Additionaly, if you want to change the mel-spec config, then go to Vocoder/melspec

## How to train the model by yourself?
In order to train the model you would need to perform simple steps, but wait for a long time for them to actually download all the data which is a LjDataset

```
chmod a+x prepare_data.sh
./prepare_data.sh
python3 train.py -c config.json -k wandb_key
```
All the other parameters are manually stored in the config.json, but you can look up the config options in train.py in order to change everything right from terminal.


## Wandb report

Here is the [link](https://wandb.ai/aamaksutova/vocoder/reports/HifiGan-project-report--Vmlldzo2MTY0NDAw?accessToken=us6u702jtkujv3cbpzyra31h96to29an4ffbbp5yo99q87ywtoii2ffrvql6lpsj) to my wandb report with all the architecture explanation and wavs + graphs

