import torch
from ultralytics import YOLO

def create_model():
    # Create a new YOLO model from scratch
    model = YOLO("yolov8n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")
    model.to('cuda')

    return model

def train_model(model):
    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data="coco8.yaml", epochs=1, batch=4, imgsz=640) 
    return model, results

def validate_model(model, results):
    # Evaluate the model's performance on the validation set
    results = model.val()
    return model, results

def detection(model, results):
    # Perform object detection on an image using the model
    results = model("animals.mp4", show=True, save=True)
    return model, results

    
def success(model):
    # Export the model to ONNX format
    success = model.export(format="onnx")
    return success

def clear_cache():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    clear_cache()
    model = create_model()

    clear_cache()
    model, results = train_model(model)

    clear_cache()
    model, results = validate_model(model, results)

    clear_cache()
    model, results = detection(model, results)

    # clear_cache()
    # s = success(model)
    
    

if __name__ == '__main__':
    main()