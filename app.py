import gradio as gr
from fastai.vision.all import *
import os

class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    split_idx, order = None, 2

    def __init__(self, train_aug, valid_aug): store_attr()
    
    def before_call(self, b, split_idx):
        self.idx = split_idx
    
    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
    
# Define model path as a string and check if the file exists
model_path = 'models/hair-resnet18-model.pkl'



# Check if model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model
learn = load_learner(model_path)

def predict_hair(img):
    try:
        pred, pred_idx, probs = learn.predict(img)
        return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}
    except Exception as e:
        return f"Error in prediction: {e}"

# Check if example paths are correct and accessible
example_paths = ["examples/1.jpg", "examples/2.jpg"]
valid_examples = []

for example_path in example_paths:
    if os.path.isfile(example_path):
        valid_examples.append(example_path)
    else:
        print(f"Example file not found: {example_path}")

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_hair,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Hair Type Classifier",
    description="A classifier to predict hair type: curly, straight, wavy, kinky.",
    examples=valid_examples  # Use only valid examples
)

# Disable Gradio analytics to prevent timeout issues
gr.Interface.analytics = False

# Launch the app
demo.launch()
