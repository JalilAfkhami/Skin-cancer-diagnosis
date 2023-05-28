from ultralytics import YOLO

class PredictProcessed():
    def predict(self):
        model = YOLO("runs/segment/train/weights/best.pt")
        model.predict(data="data/raw/test/images",
                  workers=0, 
                  device=0, 
                  save=True)

class PredictUnProcessed():
    def predict(self):
        model = YOLO("runs/segment/train2/weights/best.pt")
        model.predict(data="data/raw/test/images", 
                  workers=0, 
                  save=True)

class PredictTunedHyp():
    def predicter(self):
        model = YOLO("runs/segment/train_tunedhyp/weights/best.pt")
        model.predict(source="data/raw/test/images", 
                      save=True,
                      name="predict_tunedhyp",
                      )

if __name__ == "__main__":    
    # predicting on processed data
    # save dir = runs/segment/predict
    p_predict = PredictProcessed()
    p_predict.predict()

    # predicting on unprocessed data
    # save dir = runs/segment/predict2
    unp_predict = PredictUnProcessed()
    unp_predict.predict()

    # predicting on processed data
    # save dir = runs/segment/predict_tunedhyp
    thyp_predict = PredictTunedHyp()
    thyp_predict.predicter()