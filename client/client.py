import asyncio

import numpy as np
import streamlit as st

from config import GenRequest, default_values, model_map, pipeline_map, scheduler_map
from utils import generate_images, get_dimensions, long_poll_task_result

# Check if the session state object exists
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if "generation_in_progress" not in st.session_state:
    st.session_state.generation_in_progress = False

if "seed_value" not in st.session_state:
    st.session_state.seed_value = default_values.generator


# Function to store image in the session state
def store_image(image, prompt):
    st.session_state.generated_images.append({
        "bytes": image,
        "prompt": prompt
    })
    st.session_state.generated_images = st.session_state.generated_images[-5:]


# Function to change the generation button status: enabled/disabled
def swap_generation_button_status():
    st.session_state.generation_in_progress = not st.session_state.generation_in_progress


# Function to generate a random seed
def generate_seed():
    st.session_state.seed_value = int(np.random.randint(1, 10 ** 10))


# Streamlit layout
st.title("Image Generator with Diffusion Models")

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
prompt = st.sidebar.text_area("Prompt", default_values.prompt)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Negative Prompt
negative_prompt = st.sidebar.text_area("Negative Prompt", default_values.negative_prompt)
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

# Checkbox for Generate Random Seed
generate_random_seed = st.sidebar.checkbox("Generate Random Seed")

if generate_random_seed:
    # Generate random seed and update the input field
    seed_value = st.session_state.seed_value
    seed_value_input = seed_placeholder.number_input("Seed", value=seed_value, step=1)
    # Button to reload seed
    reload_button = st.sidebar.button("Reload Seed", key="reload_button", on_click=generate_seed)
else:
    # User input for seed value
    seed_value = seed_placeholder.number_input("Seed", min_value=1, max_value=10 ** 10,
                                               value=st.session_state.seed_value)
    seed_value_input = None

# Add a blank line in the sidebar
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# Button to trigger image generation
generation_button = st.sidebar.button("Generate Images", type="primary",
                                      on_click=swap_generation_button_status,
                                      disabled=st.session_state.generation_in_progress)

# -------------------------------------------------------------------------------------------------------
# MAIN VIEW
# -------------------------------------------------------------------------------------------------------

# Image placeholder
# Check if there are existing generated images
if st.session_state.generated_images:
    # Show the last generated image
    last_image = st.session_state.generated_images[-1]
    placeholder = st.image(last_image["bytes"], caption=prompt, use_column_width=True)
else:
    # If no existing images, show the placeholder
    placeholder = st.image("https://www.pulsecarshalton.co.uk/wp-content/uploads/2016/08/jk-placeholder-image.jpg")

# Button to trigger image generation
if generation_button:

    # Set generation in progress to True
    st.session_state.generation_in_progress = True

    # API endpoint to generate images
    api_endpoint = "http://server:8000/generate"

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

    # Clear placeholder
    placeholder.empty()

    with st.container():
        with st.spinner("Generating image..."):
            # Call the asynchronous function and wait for it to complete
            response = asyncio.run(generate_images(api_endpoint, params.model_dump()))

        if response.status_code == 200:

            # get the task id
            task_id = response.json()["taskid"]

            # API endpoint to get the result
            result_url = f"http://server:8000/results/{task_id}"

            # Call the asynchronous function and wait for it to complete
            response = asyncio.run(long_poll_task_result(task_id, result_url))

            # Display the generated image
            image = response.content
            st.image(image, caption=prompt, use_column_width=True)
            store_image(image, prompt)
            # Reset generation in progress
            swap_generation_button_status()
            # Generate a new seed if random seed is enabled
            if generate_random_seed:
                generate_seed()
        else:
            st.error(f"Error: {response.status_code}. Failed to generate images.")
            # Reset generation in progress
            swap_generation_button_status()

    st.rerun()

with st.container():
    st.markdown("----", unsafe_allow_html=True)
    st.markdown("### Previous results")
    for i, img in enumerate(st.session_state.generated_images[::-1][1:]):
        # Display image & prompt
        st.image(img["bytes"])
        st.markdown(img["prompt"])
