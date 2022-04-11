# SELAB-HCMUS MediaEval 2021 Submission

MediaEval 2021: Emotions and Themes in Music

This repo is based on the SELAB-HCMUS submission for [MediaEval 2021](https://multimediaeval.github.io/editions/2021/tasks/music/). 

Our ensemble model achieved the 4th rank at the challenge. See [results](https://multimediaeval.github.io/2021-Emotion-and-Theme-Recognition-in-Music-Task/results). Our preliminary working note is available in [this link](https://2021.multimediaeval.com/paper44.pdf).

## Requirements
See requirements.txt
```
pip install -r requirements.txt
```
## Usage
Please prepair the directory containing the preprocessed mel-spectrogram features (See [this repo](https://github.com/MTG/mtg-jamendo-dataset) for more details). Run the following command to train the model:
```
python main.py --model_dir MODEL_DIR --data_dir DATA_DIR --train TRAIN
               --valid VALID --test TEST --name NAME [NAME ...] --size SIZE
               --forget_rate FORGET_RATE --chunk_size CHUNK_SIZE --cut_size
               CUT_SIZE --batch_size BATCH_SIZE --epoch EPOCH
```
where: 
* `model_dir`: the path to the directory to save and load model
* `data_dir`: the path to data directory
* `train`, `valid`, `test`: path to .tsv file with music tag labels
* `name`: list of types of model used for training supported by timm library (for example: ResNet, MobileNet,...)
* `size`: the size of converted features
* `forget_rate`: the forget rate described in our working note
* `chunk_size`: the number of samples used for evaluation
* `cut_size`: the size of each sample to be cut for evaluation 

