from ultralytics import YOLO

class TrainProcessedData():
    def trainer(self):
        results = model.train(
            data="data/data.yaml",
            epochs=100,
            batch=2,
            imgsz=640,
            workers=0,
            device=0) # GPU

class TrainUnProcessedData():
    def trainer(self):
        results = model.train(
            data="data/raw_data.yaml",
            epochs=100,
            batch=2,
            imgsz=640,
            workers=0,
            device=0) # GPU

class TrainTunedHyp():
    def trainer(self):
        results = model.train(
            data="data/raw_data.yaml",
            epochs=100,
            batch=2,        
            imgsz=1280,     # resize to gain better mAP
            workers=0,
            name="train_tunedhyp",
            device=0) # GPU
        
if __name__ == "__main__":
    # Load a pretrained YOLO model
    model = YOLO("yolov8n-seg.pt")
    
    # Train the model on processed data      
    # save dir = runs/segment/train
    p_trainer = TrainProcessedData()
    p_trainer.trainer()
    
    # Train on raw data     
    # save dir = runs/segment/train 2
    unp_trainer = TrainUnProcessedData()
    unp_trainer.trainer()
    
    # ReTrain with Tuned Hyperparameters     
    # save dir = runs/segment/train_tunedhyp
    thyp_trainer = TrainTunedHyp()
    thyp_trainer.trainer()
    