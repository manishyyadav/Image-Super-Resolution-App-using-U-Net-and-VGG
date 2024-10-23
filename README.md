# Image-Super-Resolution-App-using-U-Net-and-VGG
Image Super-Resolution application using Streamlit to enhance low-quality images via a U-Net model integrated with VGG-based perceptual loss.
Image Super Resolution App using U-Net and VGG
This project is an Image Super-Resolution application built using Streamlit that enhances low-resolution images to higher resolution using a U-Net model integrated with a VGG-based perceptual loss. The application allows users to upload low-quality images, and then provides an enhanced image along with key image quality metrics like SSIM (Structural Similarity Index Measure), PSNR (Peak Signal-to-Noise Ratio), and MSE (Mean Squared Error).

Features
Upload Image: Users can upload low-resolution images in .jpg, .jpeg, or .png formats.
Image Enhancement: The U-Net model, trained with VGG perceptual loss, enhances the image resolution.
Quality Metrics: The app calculates and displays key metrics to compare the low-resolution and enhanced images:
SSIM: Measures the similarity between the two images.
PSNR: Evaluates the signal quality of the enhanced image.
MSE: Computes the mean squared error between the original and enhanced images.
Download Enhanced Image: Users can download the enhanced image in PNG format.
How it Works
The application uses a U-Net architecture to generate the enhanced image from the low-resolution input.
Perceptual loss is used during training, incorporating a VGG network to capture finer image details and improve visual quality.
The enhanced image is compared to the original input image, and metrics like SSIM, PSNR, and MSE are calculated to quantify the improvements.
Model Details
U-Net: The U-Net architecture is a convolutional neural network (CNN) widely used for image segmentation and enhancement. It consists of an encoder-decoder structure that captures multi-scale features of the image.
VGG-Based Perceptual Loss: VGG, a pre-trained image classification network, is used in the loss function to ensure that the enhanced image preserves perceptual features, such as edges and textures, as seen by humans.
