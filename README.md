# IPR_MAML
## Setup

1. **Prepare the Dataset**
   - Place the datasets (in pickle and CSV formats) from drive inside a `data/` folder in the directory where the repo is downloaded.

2. **Create the Environment**
   - Use the provided `environment.yml` file to create a new conda environment:
     ```bash
     conda env create -f environment.yml
     conda activate SER
     ```

## Instructions

### 1. TransferSER

1. **Update Dataset Paths**
   - In `train.py`, update the source datasets path on **line 22**.

2. **Train the Model**
   - Run the training script to obtain a pretrained model:
     ```bash
     python train.py
     ```
   - The pretrained model will be saved in the `checkpoints/` folder.

3. **Run Fractional Training**
   - Update `frac_train.py`:
     - Modify **line 24** to set `dataset_name` to the name of the test set.
     - Update **line 35** to set the model path in the `model_path` list (e.g., `"checkpoints/tess_ravdess_emodb.pt"`).
   - Run `frac_train.py`:
     ```bash
     python frac_train.py
     ```

### 2. Multi-Task Classification

1. **Update Dataset Lists**
   - In `utils.py`, modify the training and testing dataset lists on **lines 112 and 113**.

2. **Run the Script**
   - Use the following command to train or evaluate the model:
     ```bash
     python main.py --<train/eval> --save_path <save_path>
     ```
   - Replace `<train/eval>` with either `train` or `eval`, and `<save_path>` with the desired path for saving the results.

### 3. MAML (Model-Agnostic Meta-Learning)

1. **Update the Dataset**
   - In `dataloader.py`, modify **lines 13 and 14** to specify the dataset paths.

2. **Configure Parameters**
   - Edit `params.json` to adjust the parameters according to the dataset (currently configured for Shemo).

3. **Train the Model**
   - Run the training script:
     ```bash
     python train.py --model_dir <save_path>
     ```
   - Replace `<save_path>` with the desired directory for saving the model (e.g., `./`).
