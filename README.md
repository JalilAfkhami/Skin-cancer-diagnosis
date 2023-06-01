# Skin-cancer-diagnosis
Skin cancer detection using YOLOv8n-seg

## Project Structure
- src: consists of Python scripts
- data: consists of data
- runs: consists of weight and result image files
- notebook: consists of Jupyter Notebooks
## Results
The mAP[Box] at 0.50 IoU is 57.4 and at 0.50:0.95 IoU is 42.7. And the mAP[Mask] at 0.50 IoU is 55.9 and at 0.50:0.95 IoU is 41.2. Considering that it is a Nano model, it’s pretty good.

![results](https://github.com/JalilAfkhami/Skin-cancer-diagnosis/assets/111174026/02adf02b-0236-49d5-a2bb-b5f47b9c560c)

To see more details, cheack: 
*visualize_results.ipynb in notebooks*
## Set Up the Project
1. Install requirement packages
```
pip install -r requirements.txt
```
## Run the Project
- To train the model on dataset
```
python src/model/train.py
```
- To validate the model on validation dataset
```
python src/model/val.py
```
- To predict test dataset
```
python src/model/predict.py
```
