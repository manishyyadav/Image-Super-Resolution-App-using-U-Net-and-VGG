import streamlit as st
from PIL import Image
import numpy as np
import torch
from ModelUtils import Enhance
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

st.title("Image Super Resolution App")

uploaded_file = st.file_uploader("Choose a Low Quality Image...", type=["jpg", "jpeg", "png"])

weights_pth = r"C:\\New Volume\\MajorProject\\UNETlatest1.pth"  # Enter path to model weights here

def enhance_image(input_image_pth):
    output_image = Enhance(input_image_pth, weights_pth)
    return output_image

# Function to calculate SSIM with dynamic win_size adjustment
def calculate_ssim(sr_img, hr_img):
    min_side = min(sr_img.shape[0], sr_img.shape[1])
    win_size = min(min_side, 7)
    if win_size % 2 == 0:
        win_size -= 1
    return ssim(sr_img, hr_img, win_size=win_size, channel_axis=-1, data_range=1.0)

def calculate_mse(original, enhanced):
    return np.mean((original - enhanced) ** 2)

def calculate_metrics(original, enhanced):
    original = np.array(original)
    enhanced = np.array(enhanced)

    ssim_value = calculate_ssim(enhanced, original)

    if np.array_equal(original, enhanced):
        psnr_value = float('inf')  
    else:
        psnr_value = psnr(original, enhanced, data_range=255)

    mse_value = calculate_mse(original, enhanced)

    return ssim_value, psnr_value, mse_value

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    
    st.subheader("Low Quality Image")
    st.image(input_image, caption='Low Quality Image', width=256)

    # Convert the input image to a NumPy array
    input_image_np = np.array(input_image)
    
    # Enhance the image
    output_image = enhance_image(uploaded_file)

    # Normalize and prepare the enhanced image
    output_image_normalized = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    output_image_uint8 = (output_image_normalized * 255).astype(np.uint8)

    # Convert enhanced image back to a PIL image for display
    output_image_pil = Image.fromarray(np.transpose((output_image_uint8), (1, 2, 0)))

    st.subheader("Enhanced Image")
    st.image(output_image_pil, caption='Enhanced Image', width=256)

    try:
        ssim_value, psnr_value, mse_value = calculate_metrics(input_image, output_image_pil)

        st.write(f"SSIM Low Quality and Enhanced Image: {ssim_value:.4f}")
        st.write(f"PSNR Low Quality and Enhanced Image: {psnr_value:.4f} dB")
        st.write(f"MSE Low Quality and Enhanced Image: {mse_value:.4f}")
    except ValueError as e:
        st.error(str(e))

    buffered = BytesIO()
    output_image_pil.save(buffered, format="PNG")
    buffered.seek(0)

    st.download_button(
        label="Download Enhanced Image",
        data=buffered,
        file_name="enhanced_image.png",
        mime="image/png"
    )
