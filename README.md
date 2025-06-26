# Music Transformer with Curriculum Learning

The [Music Transformer](https://arxiv.org/abs/1809.04281), is a deep learning sequence model designed to generate music. This repository explores the application of **curriculum learning** to music generation, comparing curriculum learning approaches against baseline training methods. This project was done as part of the Master's thesis on Investigation of Curriculum Learning on Music Generation by Gritta Joshy and Qi Chen.

Our implementation builds upon the Transformer architecture to consider the relative distances between different elements of the sequence rather than / along with their absolute positions in the sequence. **The main focus of this project is investigating how curriculum learning can improve music generation quality compared to traditional training approaches.**

This repository contains Python scripts to preprocess MIDI data, train Music Transformers with both curriculum learning and baseline methods using PyTorch, generate MIDI files, and analyze the performance differences between training approaches.

## 📋 Table of Contents
- [Setting up](#-setting-up)
- [Curriculum Learning vs Baseline](#-curriculum-learning-vs-baseline)
- [Training Scripts](#-training-scripts)
- [Preprocess MIDI Data](#preprocess-midi-data)
- [Generate Music](#-generate-music)
- [Analysis Tools](#-analysis-tools)
- PyTorch 2.1.0
- Mido 1.2.9

## 🚀 Setting up
Clone the git repository, cd into it if necessary, and install the requirements. Then you're ready to preprocess MIDI files, as well as train and generate music with a Music Transformer.

```shell
git clone https://github.com/spectraldoy/music-transformer
cd ./music-transformer
pip install -r requirements.txt
```

## Preprocess MIDI Data
Most sequence models require a general upper limit on the length of the sequences being modeled, it being too computationally or memory expensive to handle longer sequences. So, suppose you have a directory of MIDI files at .../datapath/ (for instance, any of the folders in the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)), and would like to convert these files into an event vocabulary that can be trained on, cut these sequences to be less than or equal to an approximate maximum length, lth, and store this processed data in a single PyTorch tensor (for use with torch.utils.data.TensorDataset) at .../processed_data.pt. Running the preprocessing.py script as follows:

```shell
python preprocessing.py .../datapath/ .../processed_data.pt lth
```

will translate the MIDI files to the event vocabulary laid out in vocabulary.py, tokenize it with functionality from tokenizer.py, cut the data to approximately the specified lth, augment the dataset by a default set of pitch transpositions and stretches in time, and finally, store the sequences as a single concatenated PyTorch tensor at .../processed_data.pt. The cutting is done by randomly generating a number from 0 to lth, randomly sampling a window of that length from the sequence, and padding with pad_tokens to the maximum sequence length in the data. Pitch transpositions and factors of time stretching can also be specified when running the script from the shell (for details, run python preprocessing.py -h).

**NOTE:** This script will not work properly for multi-track MIDI files, and any other instruments will automatically be converted to piano (the reason for this is that I worked only with single-track piano MIDI for this project).

## 🎯 Curriculum Learning vs Baseline

This project's main contribution is exploring **curriculum learning** for music generation. We compare different curriculum learning strategies against baseline training methods to understand their impact on music generation quality.

### Curriculum Learning Approach
Curriculum learning trains models by gradually increasing the difficulty of training examples, starting with "easier" sequences and progressively introducing more complex ones. In our music generation context, sequence difficulty is determined by training loss - sequences with lower loss are considered easier.

### Training Methodologies

| Approach | Method | Description |
|----------|--------|-------------|
| **Baseline** | Standard Training | Traditional training on all data without difficulty ordering |
| **Curriculum Learning** | Progressive Difficulty | Training starts with easier sequences (60% or 80% curriculum threshold) |
| **Curriculum + LR** | CL with Learning Rate Scheduling | Combines curriculum learning with adaptive learning rate scheduling |

## 🚀 Training Scripts

| Script | Training Type | Epochs | Curriculum Threshold | Description |
|--------|---------------|--------|---------------------|-------------|
| `train_batch.py` | **Baseline** | 100 | N/A | Standard training without curriculum learning |
| `train.py` | **Standard** | 300 | N/A | Extended training for pretrained models |
| `traincl_finale_20.py` | **Curriculum Learning** | - | 60% | CL with 60% curriculum threshold |
| `traincl_finale_80.py` | **Curriculum Learning** | - | 80% | CL with 80% curriculum threshold |
| `traincl_learning.py` | **Curriculum + LR** | - | 60% | CL with 60% threshold + learning rate scheduling |

### Usage

**Baseline Training:**
```shell
python train_batch.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs
```

**Curriculum Learning Training:**
```shell
python traincl_finale_20.py .../preprocessed_data.pt .../ckpt_path.pt .../save_path.pt epochs
```

All training scripts support checkpointing and can be resumed with the `-l` or `--load-checkpoint` flag.

## 🎵 Generate Music
Given a trained Music Transformer's state_dict and hparams saved at .../save_path.pt, and specifying the path at which to save a generated MIDI file, .../gen_audio.mid, running the following:

```shell
python generate.py .../save_path.pt .../gen_audio.mid
```

will autoregressively greedy decode the outputs of the Music Transformer to generate a list of token_ids, convert those token_ids back to a MIDI file using functionality from tokenizer.py, and will save the output MIDI file at .../gen_audio.mid. Parameters for the MIDI generation can also be specified - 'argmax' or 'categorical' decode sampling, sampling temperature, the number of top_k samples to consider, and the approximate tempo of the generated audio (for more details, run python generate.py -h).

## 📊 Analysis Tools

The repository also includes several analysis scripts:

| Script | Purpose |
|--------|---------|
| `KLD.py` | Analyzing Kullback-Leibler Divergence |
| `plotKLD.py` | Plotting KLD and OA (Overall Accuracy) figures |
| `computed_difficulty.py` | Computing loss on all sequences and sorting them by difficulty |
| `scatter.py` | Plotting loss distribution and scatter plots for different models |
| `midi_analyzer.py` | Analyse generated music metrics |
