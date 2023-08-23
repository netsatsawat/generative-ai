import streamlit as st
import boto3
import json
import logging
from stability_sdk_sagemaker.predictor import StabilityPredictor
import sagemaker
from stability_sdk.api import GenerationRequest, GenerationResponse, TextPrompt
from PIL import Image
import io
import os
import base64

# Define logger - this will show in terminal
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger.setLevel(logging.INFO)

# Define the model
sagemaker_session = sagemaker.Session()
endpoint_name = '<your-endpoint-name>'  # insert you model endpoint here

# initialize stability model object
deployed_model = StabilityPredictor(
    endpoint_name=endpoint_name, 
    sagemaker_session=sagemaker_session
)

# Define the initial API parameters
style_preset_option = None
sampler_option = None
cfg_scale_option = 0
steps_option = 0
height = 1024  # fixed for image quality
width = 1024   # fixed for image quality
prompt = None

if "prompt" not in st.session_state:
    st.session_state["prompt"] = None

def generate_payload(prompt: str) -> GenerationRequest:
    model_request = GenerationRequest(
        text_prompts=[TextPrompt(text=prompt)],
        cfg_scale=cfg_scale_option,
        steps=steps_option,
        style_preset=style_preset_option,
        width=width,
        height=height,
        sampler=sampler_option,
    )
    return model_request


def call_endpoint(generated_request: GenerationRequest) -> GenerationResponse:
    model_response = deployed_model.predict(
        generated_request
    )
    return model_response


# Setting page title and header
st.set_page_config(page_title="Generative AI Playground", page_icon=":robot_face:")
st.markdown(
    "<h1 style='text-align: center;'>Text-to-Image Powered by SageMaker and Stability.ai 🧠</h1>", 
    unsafe_allow_html=True
)

# Add sidebar to the page
st.sidebar.title("SDXL Configuration")
with st.sidebar:
    # Selectbox
    add_selectbox = st.sidebar.selectbox(
        "Prompt examples",
        (
            "None",
            "Mountain with river in the middle of Bangkok",
            "Bangkok landscape at the top of building, temple, 8k, stunning", 
            "BMW car  in Bangkok street, 8k, reimagine, cyber-punk",
            "Thai football team lift the world cup trophy",
            "Mango sticky rice, delicious looking"
        ), 
        index=0
    )
    st.markdown('Use the above drop down box to generate **prompt** examples\n\n')
    
    style_preset_option = st.selectbox(
        "How style do you want to guide the image model towards to?",
        ("enhance", "photographic", "anime", "digital-art", "cinematic", "pixel-art", "line-art", "analog-film", "3d-model"),
    )
    
    sampler_option = st.selectbox(
        "Choose sampler to use for the diffusion process",
        ("DDIM", "DDPM", "K_EULER", "K_LMS", "K_DPM_2")
    )
    
    cfg_scale_option = st.slider(
        'How strictly the diffusion process to the prompt (higher keep your image closer to your prompt)?', 0, 35, 10
    )
    
    steps_option = st.slider(
        'Number of diffusion steps to run', 10, 150, 50
    )


# Create user input
user_prompt = st.text_input('Input the desired prompt:')

if add_selectbox != 'None' and user_prompt is not None and prompt is None:
    prompt = add_selectbox
    
else:
    prompt = user_prompt

logger.info(f'''
User has select input the prompt as {prompt}, with style_preset={style_preset_option}, 
sampler={sampler_option}, cfg_scale={cfg_scale_option}, steps={steps_option}
''')

if len(prompt) > 0:
    st.markdown(f"""
    This will show an image using **stable diffusion** of the desired *{prompt}* entered:
    """)
    # Create a spinner to show the image is being generated
    with st.spinner('Generating image based on prompt'):
        req_ = generate_payload(prompt)
        resp_ = call_endpoint(req_)
        image = resp_.artifacts[0].base64
        image_data = base64.b64decode(image.encode())
        image = Image.open(io.BytesIO(image_data))
        st.success('Generated stable diffusion model')

# Open and display the image on the site
    st.image(image)
    prompt = None
    
# Remark: there is known issue with download button where it will refresh the whole gadget, hence this has been removed
