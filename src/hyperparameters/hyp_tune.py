from ultralytics import YOLO
import ray
from ray import tune
import yaml

# Train
model = YOLO("yolov8n-seg.pt")
result = model.tune(
    data="data/raw_data.yaml",
    space={"lr0": ray.tune.uniform(1e-5, 1e-1),
           "momentum": ray.tune.uniform(0.6, 0.98)},
    train_args={"epochs": 10,
                "name":"hyp_tune_train"}
)
best_config = result.get_best_result(metric = 'metrics/mAP50(B)', mode = 'max')
print(best_config)

with open('runs/hyp_tune/best_config.yml', 'w') as outfile:
    yaml.dump(best_config, outfile, default_flow_style=False)
    
# Val
# Load the trained weights and validating   # on unprocessed data 
model = YOLO("runs/segment/train2/weights/best.pt")
model.val(data="data/raw_data.yaml", name="hyp_tune_val", workers=0, device=0, save=True)

# Predict
# Load the trained weights and predict      # on unprocessed data
model = YOLO("runs/segment/train2/weights/best.pt")
results = model.predict(source="data/raw/test/images", name="hyp_tune_predict", save=True)