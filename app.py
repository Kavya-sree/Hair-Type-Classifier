import gradio as gr
from fastai.vision.all import *

# Path to the saved model
model_path = Path('models\hair-resnet18-model.pkl')

# Load the model
learn = load_learner(model_path)

# Function to make predictions
def predict_hair(img):
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_hair,
    inputs=gr.Image(type="pil"),  # Removed 'shape' and replaced it with 'type="pil"'
    outputs=gr.Label(num_top_classes=3),
    title="Hair Type Classifier",
    description="A classifier to predict hair type: curly, straight, wavy, kinky.",
    examples=["examples/0a62ef878341056d842dbde6365f9c4a4.jpg", "examples/58baa8efdf105f1591f34dd7299538294.jpg"]
)

# Launch the app
demo.launch(share=True)
