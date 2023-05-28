from ultralytics import YOLO

class ValidProcessed():
    def valid(self):
        model = YOLO("runs/segment/train/weights/best.pt")
        model.val(data="data/data.yaml",
                  workers=0, 
                  device=0, 
                  save=True)

class ValidUnProcessed():
    def valid(self):
        model = YOLO("runs/segment/train2/weights/best.pt")
        model.val(data="data/raw_data.yaml", 
                  workers=0, 
                  device=0, 
                  save=True)

class ValidTunedHyp():
    def valid(self):
        model = YOLO("runs/segment/train_tunedhyp/weights/best.pt")
        model.val(data="data/raw_data.yaml",
                  workers=0, 
                  name="val_tunedhyp",
                  save=True)


if __name__ == "__main__":    
    # validating on processed data    
    # save dir = runs/segment/val
    p_val = ValidProcessed()
    p_val.valid()
    
    # validating on unprocessed data    
    # save dir = runs/segment/val2
    unp_val = ValidProcessed()
    unp_val.valid()
    
    # validating on unprocessed data    
    # save dir = runs/segment/val_tunedhyp
    thyp_val = ValidProcessed()
    thyp_val.valid()