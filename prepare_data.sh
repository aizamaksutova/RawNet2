mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

export PATH=/users/aizam/.local/bin${PATH:+:${PATH}}

kaggle datasets download awsaf49/asvpoof-2019-dataset/

mkdir data
mv asvpoof-2019-dataset.zip data/
unzip data/asvpoof-2019-dataset.zip

mv LA data/
mv PA data/