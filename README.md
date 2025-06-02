# Uni features extraction from all (detected) tissue tiles

This repository contains the code for extract UNI features from all (detected) tissue tiles. 
This package implement the features extractor described in [1] 

## Installation

To utilise the package, you need to download the code from this repository. You can do this by using the following command:

```bash
git clone https://github.com/GeNeHetX/
```

Then, you need to install the required packages. 
First, having OpenSlide is mandatory. Considerer visiting [their website](https://openslide.org/download/). In short for Linux users, choose the command corresponding to your distro. For windows users, download the Windows 64-bit Binaries and follow [these insctructions](https://openslide.org/api/python/). You will have to change the value of `OPENSLIDE_PATH` in python files. I suggest using Linux or WSL.
Then to install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```

### Getting model access
Request access to the model weights from the Huggingface model page at: https://huggingface.co/mahmoodlab/UNI.

## Usage
### Extract neoplasic features
To use the model, you can use the following code:

```bash
usage: python process_wsi.py [-h] --temp_dir TEMP_DIR [--wsi WSI] [--model_path MODEL] [--device {cuda:0,cpu,mps}]
                      [--batch_size BATCH_SIZE] [--num_workers NUM_WORKER]
```
or example with a loop
```bash
for svs_file in $(ls -r ./IMAGES/*_HE.); do
    base_name=$(basename "$svs_file" .)
    python ./UNI_extractor_All_Tiles/src/UNI_extractor/process_wsi.py --temp_dir feats --wsi "$svs_file" --model_path pytorch_model.bin --device cuda:0 --batch_size 64 --num_workers 16; done
    
```

Where:
- `--temp_dir` is the directory where the temporary files will be stored.
- `--wsi` is the path to the WSI. It accepts ".svs", ".ndpi" and ".qptiff" files. More formats can be added in the `extract_features.py` and `extract_tiles.py` files.
- `--model_path` is the path to the pretrained uni extractor features model `pytorch_model.bin`.
- `--device` is the device to use for the prediction. It can be "cuda", "mps" if you work on macOS or "cpu" if you don't have a GPU. default is "cuda".
- `--batch_size` is the batch size to use for the prediction.
- `--num_workers` is the number of workers to use for the prediction. If on Windows, it should be set to 0.


For example, you can use the following command to predict a WSI:

```bash
 python3.12 src/UNI_extractor/process_wsi.py --temp_dir data/ --wsi data/Cas02.svs --model_path data/pytorch_model.bin --pred_tiles data/tile_scores.csv --device cuda:0

```





## References
[1] Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others « Towards a General-Purpose Foundation Model for Computational Pathology ». Nature Medicine (2024) https://doi.org/10.1038/s41591-024-02857-3.


