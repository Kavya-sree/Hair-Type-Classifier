import gradio as gr
from fastai.vision.all import *

MODELS_PATH = Path('./models')
EXAMPLES_PATH = Path('./examples')

    
LEARN = load_learner(MODELS_PATH/'hair-resnet18-model.pkl')
LABELS = LEARN.dls.vocab

def predict_hair(img):
    #img = PILImage.create(img)
    pred,pred_idx,probs = LEARN.predict(img)
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

demo = gr.Interface(fn=predict_hair,
                    inputs=gr.inputs.Image(shape=(256,256)),
                    outputs=gr.outputs.Label(num_top_classes=3),
                    title="Hair Type Classifier",
                    description="A hair type classifier to predict hair type: straight, wavy, curly, kinky and dreadlocks. Although dreadlocks are not hair type, we can still classify",
                    examples= [f'{EXAMPLES_PATH}/{f.name}' for f in EXAMPLES_PATH.iterdir()],
                    )
demo.launch(enable_queue= True)
