from wo import turn_off_warnings

turn_off_warnings()

from inference import get_model
import cv2

# define the image url to use for inference
image_file = "Pictures/pic_0.png"
image = cv2.imread(image_file)

# load a pre-trained rfdetr model
model = get_model(model_id="", api_key="")

def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def get_labels(items: list, batch_size = 1):
    output = []

    for batched_items in chunked(items, batch_size):
        results = model.infer(batched_items)

        for result in results:
            output.append(result.predictions)

    return output

def get_r(img):
    return model.infer(img)