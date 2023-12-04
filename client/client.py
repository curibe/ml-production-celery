import asyncio

import numpy as np
import streamlit as st

from config import GenRequest, default_values, model_map, pipeline_map, scheduler_map
from utils import generate_images, get_dimensions

# Custom CSS style to hide the default increment/decrement buttons
custom_css = """
    <style>
        .streamlit-widget st-number-input .decrement,
        .streamlit-widget st-number-input .increment {
            display: none !important;
        }
    </style>
"""

# Streamlit layout
st.title("Diffusion Models Image Generator")

# -------------------------------------------------------------------------------------------------------
# SIDE BAR
# -------------------------------------------------------------------------------------------------------

st.sidebar.header("Model")

# Model
model_list = list(model_map.keys())
model_selected = st.sidebar.selectbox("Model", model_list,
                                      index=model_list.index("Stable Diffusion XL"))
model = model_map[model_selected]
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Scheduler
schedulers_list = list(scheduler_map.keys())
scheduler_option = st.sidebar.selectbox("Scheduler", schedulers_list,
                                        index=schedulers_list.index("PNDM"))
scheduler = scheduler_map[scheduler_option]
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Pipeline
pipeline = pipeline_map[model]

st.sidebar.header("Parameters")

# Prompt
prompt = st.sidebar.text_input("Prompt", default_values.prompt)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Negative Prompt
negative_prompt = st.sidebar.text_input("Negative Prompt", default_values.negative_prompt)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Number of Images
num_images_per_prompt = 1

# Image Size
size = st.sidebar.radio("Size", ("512x512", "768x768"), index=1)
witdh, height = get_dimensions(size)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Number of Inference Steps
num_inference_steps = st.sidebar.slider("Number of Inference Steps", min_value=1, max_value=200,
                                        value=default_values.num_inference_steps, step=1)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Guidance Scale
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=0., max_value=20., value=default_values.guidance_scale,
                                   step=0.5, format="%.1f")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# ********** Seed Options **********#
st.sidebar.markdown("#### Seed Options")

# Placeholder for dynamic UI update
seed_placeholder = st.sidebar.empty()

# Custom HTML to inject CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Checkbox for Generate Random Seed
generate_random_seed = st.sidebar.checkbox("Generate Random Seed")

if generate_random_seed:
    # Generate random seed and update the input field
    seed_value = int(np.random.randint(1, 10 ** 10))
    seed_value_input = seed_placeholder.number_input("Seed", value=seed_value, step=1)
    # Custom button with a reload icon
    reload_button = st.sidebar.button("Reload Seed", key="reload_button")
else:
    # User input for seed value
    seed_value = seed_placeholder.number_input("Seed", min_value=1, max_value=10 ** 10, value=default_values.generator)
    seed_value_input = None

# Add a blank line in the sidebar
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------------
# MAIN VIEW
# -------------------------------------------------------------------------------------------------------

# Button to trigger image generation
if st.sidebar.button("Generate Images", type="primary"):

    # API endpoint to generate images
    api_endpoint = "https://server/generate"

    # Parameters to be sent to the API
    params = GenRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=witdh,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=seed_value,
        pipeline=pipeline,
        scheduler=scheduler,
        modelid=model
    )

    # Call the asynchronous function and wait for it to complete
    response = asyncio.run(generate_images(api_endpoint, params.dict()))

    # Check if the request was successful
    if response.status_code == 200:
        # Display the generated images
        images = response.json()["images"]
        for img in images:
            st.image(img, caption="Generated Image", use_column_width=True)
    else:
        st.error(f"Error: {response.status_code}. Failed to generate images.")

# You can add additional sections or features as needed in the central part of the dashboard.
# For example, you might want to include a section to display the diffusion model parameters,
# or any other relevant information.
