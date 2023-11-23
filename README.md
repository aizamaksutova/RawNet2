# ASR Project
ASR Project within a course 'Deep Learning in Audio'

# How to recreate the experiment

## Crucial information

In [ctc_char_encoder.py](https://github.com/aizamaksutova/DL_Audio/blob/branch1/hw_asr/text_encoder/ctc_char_text_encoder.py) I added the beam search with and without LM.

### What do we need for LM?
```
pip install https://github.com/kpu/kenlm/archive/master.zip
wget https://www.openslr.org/resources/11/3-gram.arpa.gz --no-check-certificate
gzip -d 3-gram.arpa.gz
```
### How do I download the model?
```
wget https://drive.google.com/file/d/1FfcDs004kl3bo8prP-TmvpES9igAayBl/view?usp=sharing
```

### How do I run test?

For test-clean:

```
python3 test.py \
   --config hw_asr/configs/test_clean.json \
   --resume model_best.pth \
   --batch-size 64 \
   --jobs 4 \
   --beam-size 100 \
   --output output_clean.json
```

For test-other:

```
python3 test.py \
   --config hw_asr/configs/test_other.json \
   --resume model_best.pth \
   --batch-size 64 \
   --jobs 4 \
   --beam-size 100 \
   --output output_other.json
```

### How do I train the model?

Dependent on which model you want, but I trained the best model using this command
```
python3 train.py -c hw_asr/configs/train_together.json
```

### How do I evaluate on test data?
```
python3 test.py \
   -c hw_asr/configs/test_other.json \
   -r model_best.pth \
   -t test_data \
   -o test_result.json \
   -b 5
```

# Device

GPU: Tesla P100-PCIE


# Model choice 

For my implementation of ASR, I chose the model from paper [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf).
Model architecture in the project is heavily dependant on this architecture, though a bit reduced due to resources limitations. 

![Model architecture from paper Deep Speech 2](https://github.com/aizamaksutova/DL_Audio/blob/main/images/model-arch-ds2.png)

#### Specifications
For padding, kernel sizes and strides in the convolutional layers I looked up the official documentation of [DeepSpeech2](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html) in order to build a strong model from the very first steps.  

# Training pipeline

## First method

1. First I am training my model only on LibriSpeech dataset train-clean-100 and train-clean-360 for 50 epochs and testing the model quality on LibriSpeech dataset test-clean. [since the audios are clean i did not add any noise augmentations]
2. Then I am fine-tuning the model on LibriSpeech dataset train-other-500 for 50 epochs and testing its final quality on test-other. [on the fine-tuning step I am adding noise augmentations to upsample the training data and adapt the model to noisy sounds]

## Second method
The hypothesis here is that there is a decent shift in the distributions of data when we move from the clean dataset to 'other', so the first method with finetuning + pretraining would not work well. Due to that, there is a need to train on both datasets and make the model train on various examples. So the pipeline is:

1. Train on both clean + other train and eval on test-other straight away to understand the real accuracy of my model while training. 


# Experiments
 ### First try
 First I implemented a model with 2 convolution layers and 4 GRU layers. Optimizer - SGD, lr_scheduler - OneCycleLR, no augmentations for train-clean part. See the config for the clean-part training [here](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/1exp_train_clean.json). 
 The metrics on this step while training only on clean part were CER: 0.39 and I would like to not state the WER, because I was not computing it in the right way.

 ### Second try
 I decided to carry on with training on the clean part and get better cer and wer metrics on them, so I am adding more GRU layers and augmentations. I decided to take the architecture which is shown in the picture in part 1 [3 convolution layers and 7 GRU layers]. First for 50 epochs I trained on the train-clean datasets adding such augmentations: for waves - [Shift](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/shift.py), [Gain](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/gain.py) and [Guassian Noise](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/add_gaussian_noise.py); for spectrograms: [Frequency masking](https://pytorch.org/audio/main/generated/torchaudio.transforms.FrequencyMasking.html) and [Time masking](https://pytorch.org/audio/main/generated/torchaudio.transforms.TimeMasking.html). Afterwards, I trained the same model for 50 epochs on train-other with the same augmentations, but without adding the Guassian Noise. Configs for training -  [first 50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/secondexp_firstiteration.json) and second [50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/secondexp_seconditer.json). Logs of training: [first 50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/training_logs/second_exp_firstiter_train50.log) and [second 50 epochs](https://github.com/aizamaksutova/DL_Audio/blob/main/training_logs/second_exp_seconditer_train.log). 
 
#### Metrics:
[model numbers were first train_ds2/1021_175629, second train_ds2_other/1022_173704]

##### test-clean
 | Beamsearch  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.45  |    0.16 | 
##### test-other
 | Beamsearch  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.63  |    0.27 | 

#### What failed? 
Using TimeInversion Augmentation was a failure, the audio became impossible to listen to and this augmentation was rather vague.

### Third try
 
 I decided to try to train the model for 100 epochs on the whole dataset(clean + other) and see how well that goes. Model architecture was a bit changed since i had to downsize the number of GRU layers and make it only 4 due to enlargening the dataset. This attempt rather failed too [see metrics]. [logs](https://github.com/aizamaksutova/DL_Audio/blob/main/training_logs/third_alltrain.log) and [config](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/train_all.json)

 #### Metrics
##### test-clean
 | Beamsearch  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.99  |    0.71 | 
##### test-other
 | Beamsearch  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 1.02  |    0.68 | 

 #### Why failed?
 My hypothesis is that using Guassian noise augmentation with the "other" dataset was an overkill. Also, SGD optimizer might be less efficient than Adam in this case, but I wanted to try the CyclirLR scheduler with triangular2 mode and it is compatible only with SGD.
 
### Beam search without LM

You can see the code of my beamsearch without using Language model in the /text_encoder/ctc_char_text_encoder.py file, yet I wouldn't say that custom beam search actually did something valuable here, CER and WER metrics went up with beam search results [example below]

#### Beam Search [without LM] results example on the model we got from the first experiment

| Beamsearch  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.46  |    0.16 | 
| YES  | 0.47  |   0.17 |

[weird, no rise in metrics]


 ### Fourth try

 Here we have some hyperparameters games, didn't change much though - only changed the number of epochs + removed the guassian noise augmentation + we are still training on clean + other train, but now in one epoch we have 1000 steps [might be an overkill] + running val on test-other to see how good of a score we can get. Since the training data is large, i removed the number of GRU layers[left 4 GRU layers and 2 convolutional layers] and got left with only 19M params - this might result in insufficient quality.

 #### Metrics

 1. Since one epoch lasted for about 20 minutes due to the number of steps per epoch - I decided to stop the training on the 33 epoch to see how well this change in hyperparameters resulted in the model accuracy.
On test-clean:

| Beamsearch with LM  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.27  |    0.08 | 
| YES  | 0.16  |   0.06 |

##### This is a solid 7

On test-other:

| Beamsearch with LM  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.45  |    0.19 | 
| YES  | 0.46  |   0.19 |



 ### Beam search with LM

#### Implementation
For beam search with LM i used the 3-gram.arpa LM from LibriSpeech models, implemented a ctc decoder which is able to decode texts. Additionally, I added some hot words[hard words which not a lot of people could guess how to write] which are passed to the decoder with a significant weight for the model to draw attention to these words specifically. 

#### Hyperparameters 
In the prior experiment[№4] I implemented Beam Search with LM with hyperparameters alpha = 0.5 and beta = 0.1, maybe this needs some changes so I will experiment with these in the fifth, final experiment.

# Best model

#### Download the model

You can download the model's weights [here](https://drive.google.com/file/d/1FfcDs004kl3bo8prP-TmvpES9igAayBl/view?usp=sharing)

#### Architecture 

2 Convolutional layers + 5 GRU layers. In the paper [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf) authors said that there is no radical gain in 3 convolutional layers and 7 GRU layers, and one can stop on 2 conv layers + 4 gru layers, but I decided to add one more just in case(it didn't slow down the training process too much). 

#### Augmentations
For waves I used Shift, Gain, Guassian Noise[links to their docs are given above] and for spectrograms I used TimeMasking and FreqMasking. Since the dataset i was training on was clean + other, I decided that the probablitity of adding Guassian Noise should not be 1 as in other experiments, because it can make the data too inaudible. You can listen to the audio + look at spectrograms after augmentations [here](https://wandb.ai/aamaksutova/asr_project/reports/Best-model-report--Vmlldzo1ODAwMjA1?accessToken=57lxhs8y0bv0x8v931lhjojd75yh9e601ots16d4jdra324x4740n8w269o1cyjl).

#### Metrics


On test-clean:

| Beamsearch with LM  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.22  |    0.069 | 
| YES  | 0.1996  |   0.074 |


On test-other:

| Beamsearch with LM  | WER |  CER  |
| ------------- | ------------- | ------------- | 
| NO  | 0.43  |    0.17 | 
| YES  | 0.42  |   0.198 |

#### logs

Config for [train](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/final_exp.json), for [test-clean](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/test-clean.json) and [test-other](https://github.com/aizamaksutova/DL_Audio/blob/main/configs/test-other.json). Training logs are [here](https://github.com/aizamaksutova/DL_Audio/blob/main/training_logs/final_train.log)

#### encoder

For LM I tried different hyperparameters for alpha and beta, but the best ones were alpha = 0.5 and beta = 0.1, yet i tried different betas from 1e-3 to 0.1


# Wandb reports with all the necessary data
#### for the best run
[Here](https://wandb.ai/aamaksutova/asr_project/reports/Best-model-report--Vmlldzo1ODAwMjA1?accessToken=57lxhs8y0bv0x8v931lhjojd75yh9e601ots16d4jdra324x4740n8w269o1cyjl)

#### for all the good runs
[Here](https://wandb.ai/aamaksutova/asr_project/reports/Report-for-all-the-runs--Vmlldzo1ODAwNTk2?accessToken=lpmas2p2a1a25ne4ndzl97473at90pgkzs0jq187kmt7xl1yp4bgfce2wgtfbbv7)
