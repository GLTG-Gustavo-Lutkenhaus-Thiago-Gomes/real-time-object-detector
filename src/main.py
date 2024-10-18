import torch
from ultralytics import YOLO
from roboflow import Roboflow
import cv2

def load_model():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("C:\\Users\\gomes\\Documents\\yolo\\real-time-object-detector\\src\\yolov8n.pt")
    model.to('cuda')

    return model

def train_model(model):
    results = model.train(data="./data.yaml", epochs=9000, batch=0.80, imgsz=640, save_period=100, augment=True, patience=0)
    model.save('yolov8n.pt')

    return model, results

def validate_model(model, results):
    results = model.val()

    return model, results

def detection(model, results):
    results = model("C:\\Users\\gomes\\Documents\\yolo\\real-time-object-detector\\images", show=True, save=True)

    return model, results

def detection_only(model):
    results = model("C:\\Users\\gomes\\Documents\\yolo\\real-time-object-detector\\images")
    cont = 0
    for result in results:
        boxes = result.boxes  
        masks = result.masks  
        keypoints = result.keypoints  
        probs = result.probs  
        obb = result.obb  
        # result.show()  
        result.save(filename="C:\\Users\\gomes\\Documents\\yolo\\real-time-object-detector\\result_images\\"+str(cont)+"_result.jpg")  # save to disk
        cont = cont + 1

    return results
    
# def success(model):
#     success = model.export(format="onnx")
#     return success

def clear_cache():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    clear_cache()
    model = load_model()

    # clear_cache()
    # model, results = train_model(model)

    # clear_cache()
    # model, results = validate_model(model, results)

    # clear_cache()
    # model, results = detection(model, results)

    clear_cache()
    _ = detection_only(model)


if __name__ == '__main__':
    main()