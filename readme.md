# Basic Classifier

This repo contains an example classifier made with pytorch.

## Setup

```bash
pip3 install -r requirements.txt
```

Update the *data_path* and *output_path* values in the *train.py* script.

## Dataset

You can use any classification dataset; the *dataset.py* script expects it to be in the form

```
| dataset_root /
| - train/
| -- class 1/
| --- img1
| --- img2
| --- ...
| -- class 2/
| --- ...
| -- class n/
| --- ...
| - val/
| -- class 1/
| --- img1
| --- img2
| --- ...
| -- class 2/
| --- ...
| -- class n/
```

meaning that you have already split training and validation subsets.

## Usage

Run with:

```bash
python3 train.py
```
