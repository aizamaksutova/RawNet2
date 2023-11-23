# FastSpeech2
Project on text-to-speech. This rep contains my implementation of FastSpeech2 model and all the steps to reimplement the pipeline

## How to install?

Make sure to follow this guide
```
git clone https://github.com/aizamaksutova/FastSpeech2.git
cd FastSpeech2
pip install -r requirements.txt
```

## How to inference?

First, you should create a file (e.g. texts.txt) with all the phrases you want to reproduce as wavs. The format of .txt file is the same as the file inference.txt

Then perform these:

```
#downloading all the needed models
chmod a+x prepare_inf.sh
./prepare_inf.sh
python3 test.py -c config.json -r final_model.pth -f inference.txt
```
Afterwards, you will see the results in the results/ directory
## How to train the model by yourself?
In order to train the model you would need to perform simple steps, but wait for a long time for them to actually download all the data + manually perform all the pitch and energy in advance

```
chmod a+x prepare_data.sh
./prepare_data.sh
python3 prepare_pitch-energy.py

```
