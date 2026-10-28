# PlantDoc AI

Plant leaf disease classifier built with PyTorch. Upload a photo of a leaf and the model predicts one of 38 classes (14 plants, healthy or diseased), with a confidence score.

The model is a convolutional neural network trained from scratch, without pretrained weights, on the [Balanced PlantVillage dataset](https://www.kaggle.com/datasets/kanivan2203/balanced-plantvillage-dataset).

## Results

Measured on 6,649 held-out test images that the model never saw during training or model selection:

| Metric | Value |
|---|---|
| Test accuracy | 99.53% |
| Macro F1 | 0.994 |
| Test loss | 0.0189 |
| Wrong predictions | 31 of 6,649 |

The most common mistake is Corn Northern Leaf Blight predicted as Corn Gray Leaf Spot (6 images). Both diseases cause long gray-tan lesions.

Trained on a Mac mini M4 using the Apple GPU (MPS). The best model came from epoch 56.

## How it works

1. `download_data.py` downloads the dataset from Kaggle into `data/plantvillage/`.
2. `src/dataset.py` scans every image, skips unreadable files, labels each image from its folder, and finds exact and near-duplicate images. It splits the data 70/15/15 into train, validation and test, stratified by class, and keeps duplicates together so no test image has a copy in training. The split is saved to `outputs/splits/splits.csv` and reused by every script.
3. `train.py` trains the CNN with data augmentation (random crops, flips, rotation, color changes), early stopping, learning-rate reduction, and saves the model with the lowest validation loss.
4. `evaluate.py` scores the saved model on the test set and writes reports.
5. `predict.py` and `app.py` load the model and preprocess images with the same settings saved during training.

### Dataset layout and cleaning

The Kaggle dataset is organized for a two-stage system: a `stage2` folder labels leaves by plant only (for example `Apple`), and `stage3_<plant>` folders label diseases for each plant. A plant-only folder mixes healthy and diseased leaves under one name, so the model can't learn it.

`dataset.py` removes plant-only classes when disease classes exist for that plant. It keeps Blueberry, Orange, Raspberry, Soybean and Squash, which only appear at plant level because PlantVillage has one class for each of them. The first training run kept all 45 classes and reached 90.8% test accuracy, with at least 570 of its 678 mistakes involving the plant-only classes. Removing them brought the dataset to 38 classes and 99.5% accuracy.

### Model

Five convolutional blocks (each: two 3×3 convolutions with BatchNorm and ReLU, then 2×2 max pooling), going from 32 to 256 channels, followed by global average pooling, dropout (0.3) and a linear output layer. About 2.4 million trainable parameters. Input images are 128×128.

## Setup

Requires Python 3.10 to 3.12. Newer Python versions may not have PyTorch builds yet.

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

On Apple Silicon Macs, PyTorch uses the GPU automatically. On other machines it uses an NVIDIA GPU if one is available, otherwise the CPU.

## Dataset

```bash
python download_data.py
```

This downloads the dataset with `kagglehub` and copies it into `data/plantvillage/`. If Kaggle asks you to log in, create an API token at kaggle.com under Settings → API, and save the `kaggle.json` file to `~/.kaggle/`.

To use a dataset stored somewhere else, set `DATASET_DIR` in `.env`. Relative paths are resolved from the project folder.

Check the classes and split before training:

```bash
python -m src.dataset
```

The first run checks every image and takes under a minute on an M4. If you change `EXCLUDE_DIRS` or the split fractions in `.env`, delete `outputs/splits/` so the split is rebuilt.

## Training

```bash
python train.py
```

Settings such as epochs, batch size, learning rate and image size are in `.env`. The run used for the results above had `EPOCHS=60`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001` and `IMAGE_SIZE=128`. Press Ctrl+C to stop early. The best model so far is kept.

Outputs:
- `models/plantdoc_cnn.pt` stores the weights, class names, image size and normalization values
- `models/class_names.json` maps model outputs to class names
- `outputs/training_curves.png` and `outputs/training_history.csv`

## Evaluation

```bash
python evaluate.py
```

Writes to `outputs/`: `test_summary.json`, `classification_report.txt`, `confusion_matrix.png`, `confused_pairs.csv`, and `wrong_predictions.csv` (every misclassified test image, most confident mistakes first).

## Predicting

From the terminal:

```bash
python predict.py path/to/leaf.jpg
python predict.py --random-test     # random test image, also shows its true class
```

Web app:

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Upload a JPG or PNG, take a photo with your camera, or pick a random test-set image. Each result shows:

- the plant, condition, cause (scientific name), type (fungal, bacterial, viral, pest) and typical impact
- confidence, the runner-up class, and the top predictions (3 to 10, set in the sidebar)
- symptoms, what makes it spread, treatment and prevention (for healthy leaves: care tips and what to watch for)
- a Grad-CAM heatmap of the parts of the photo that drove the prediction
- how reliable that class was on the test set (precision, recall, and which classes were mistaken for it)
- photo checks for resolution, brightness, blur and how much of the frame the leaf fills
- a downloadable Markdown report

The disease information lives in `src/disease_info.py`. It is general guidance; check product labels and local advice before treating plants.

## Reading the confidence score

The confidence is the model's probability for a class. Probabilities across all 38 classes add up to 100%.

- **Above 90%:** the image closely matches what the model learned for that class.
- **60 to 90%:** probably right, but check the second prediction, especially if it's the same plant.
- **Below 60%:** the app flags the result as uncertain. The photo may be unclear, or it may show a plant or problem the model doesn't know.

A high score is not proof. The model always picks one of its 38 classes, even for a photo of something it has never seen, and it can be confidently wrong on such images.

## Limitations

- **Lab photos.** PlantVillage images show single leaves on plain backgrounds under even lighting. The 99.5% accuracy applies to that kind of image. Photos from a real field, with several leaves, soil, shadows or other backgrounds, will usually score lower.
- **Fixed set of classes.** The model only knows 14 plants and 38 classes. Any other plant or disease gets labeled as one of these.
- **Single-class plants.** Blueberry, Raspberry and Soybean only have healthy images, and Orange and Squash only have one disease each. For these plants the model learned what the plant looks like, not what the disease looks like. A diseased soybean leaf would most likely be called healthy.
- **Duplicate detection** catches identical files and near-identical images (same perceptual hash). Photos of the same leaf taken from clearly different angles could still end up in different splits.
- **One leaf per image.** The model classifies the whole image and doesn't locate disease spots.
- Predictions are estimates and do not replace diagnosis by an agricultural expert.

## Project structure

```
PlantDoc-AI/
├── app.py               Streamlit web app
├── train.py             training loop
├── evaluate.py          test-set evaluation and reports
├── predict.py           single-image prediction (also used by the app)
├── download_data.py     downloads the dataset from Kaggle
├── inspect_data.py      lists image folders and counts
├── config.py            settings, read from .env
├── requirements.txt
├── .env.example
├── src/
│   ├── dataset.py       scanning, cleaning, splitting, DataLoaders
│   ├── model.py         CNN definition, saving and loading
│   ├── disease_info.py  symptoms, causes, treatment and prevention for each class
│   ├── insights.py      photo checks, heatmap overlay, per-class reliability, reports
│   └── utils.py         seeds, device, plots, class names
├── data/plantvillage/   dataset (not in git)
├── models/              trained model (not in git)
└── outputs/             splits, reports, plots (not in git)
```# PlantDoc-AI
