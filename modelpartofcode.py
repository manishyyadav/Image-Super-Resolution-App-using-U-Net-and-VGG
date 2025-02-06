
# %%
class DoubleConv(nn.Module):
    
    def __init__(self, in_channels,out_channels):
        
        super().__init__()
        
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), #we're preserving input height and width
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), #we're preserving input height and width
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)

# %%
class UNET(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512,1024]):
        
        super().__init__()
        
        self.downsamples=nn.ModuleList()
        
        self.upsamples= nn.ModuleList()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #now building the encoder
        
        for feature in features:
            self.downsamples.append(DoubleConv(in_channels, feature))
            in_channels=feature

        #the decoder
        
        for feature in reversed(features):
            self.upsamples.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.upsamples.append(DoubleConv(feature*2, feature))
            
        #bottleneck layer
        self.bottleneck= DoubleConv(features[-1], features[-1]*2)

        #final 1x1 conv to change out_channels
        self.final= nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        
    def forward(self,x):
        skip_connections=[]

        for downsampler in self.downsamples:

            x = downsampler(x) #this is the output of each downsampling layer
            skip_connections.append(x)
            x= self.pool(x)

        x= self.bottleneck(x) 

        skip_connections= skip_connections[::-1]

        for i in range(0, len(self.upsamples), 2):

            x= self.upsamples[i](x) # this is the conv transpose layer
            skipped= skip_connections[i//2]

            concat_skipped= torch.cat((skipped,x),dim=1)

            x=self.upsamples[i+1](concat_skipped) #this is the double conv layer

        return self.final(x)

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %%
torch.manual_seed(42)

model = UNET().to(device)

x = torch.randn(1, 3, 256, 256).to(device)
    
output = model(x)

print("Final output size:", output.size())

# %%
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params

# %% [markdown]
# ## VGG Loss

# %%
import torch
import torchvision.models as models

# Specify the path to the uploaded VGG16 weights file
model_path = '/kaggle/input/vgg/pytorch/default/1/vgg16-397923af.pth'  # Replace 'your-dataset-name' with the actual name

# Load the VGG16 model architecture
vgg = models.vgg16()

# Load the local weights into the model
vgg.load_state_dict(torch.load(model_path))



# %%
class VGGPerceptualLoss(nn.Module):


    
    def __init__(self, feature_extractor, diction, criterion, lambdas):
        super(VGGPerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.diction = diction
        self.criterion = criterion
        self.lambdas = lambdas

    def forward(self, x, y):
        x_features = self.feature_extractor(x,self.diction)
        y_features = self.feature_extractor(y,self.diction)
#         print(len(x_features), len(y_features))
        loss = 0
        for layer_name,_ in self.diction.items():
            x_feat = x_features[layer_name]
            y_feat = y_features[layer_name]
            
            loss += self.lambdas[layer_name] * self.criterion(x_feat, y_feat)        
        return loss


# %%
features = list(vgg.features.children())[:28]
features = nn.Sequential(*features)
layers = {'6':'6','15':'15','27':'27'}
vgg_feature_extractor = {'6': features[:7].to(device), '15': features[7:16].to(device), '27': features[16:].to(device)}
lambdas = {'6': 0.3, '15': 0.3, '27': 0.4}
features

# %%
def feature_extractor(x, dic):
    
    features = {}
    for name, sequence in dic.items():
        x= sequence(x)
        features[name] = x
    
    return features   

# %%
criterion_mse = nn.MSELoss()

vgg_tester= VGGPerceptualLoss(feature_extractor, vgg_feature_extractor, criterion_mse,lambdas)

# %%
rand1 = torch.rand(8, 3, 256, 256).to(device)
rand2= torch.rand(8,3,256,256).to(device)

# %%
vgg_loss= vgg_tester.forward(highres[0].to(device), highres[0].to(device))
print(type(vgg_loss), vgg_loss)

# %% [markdown]
# ## Training Loop

# %%
criterion = nn.MSELoss() #
criterion_vgg = VGGPerceptualLoss(feature_extractor, vgg_feature_extractor, criterion_mse,lambdas).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# %%
from skimage.metrics import structural_similarity as ssim

# Your function to calculate SSIM with win_size adjustment
def calculate_ssim(sr_img, hr_img):
    # Ensure win_size is less than or equal to the smallest side of the image
    win_size = min(sr_img.shape[0], sr_img.shape[1], 7)  # Example: win_size = 7 if images are >= 7x7
    return ssim(sr_img, hr_img, win_size=win_size, multichannel=True)


# %%
num_epochs = 14

# %%
def calculate_ssim(sr_img, hr_img):
    # Find the smallest dimension of the image
    min_side = min(sr_img.shape[0], sr_img.shape[1])
    
    # Ensure win_size is an odd number and less than or equal to min_side
    win_size = min(min_side, 7)  # Choose the smaller value between min_side and 7
    if win_size % 2 == 0:  # Ensure win_size is odd
        win_size -= 1
    
    # Calculate SSIM, setting channel_axis for RGB images and data_range for floating point images
    return ssim(sr_img, hr_img, win_size=win_size, channel_axis=-1, data_range=1.0)


# %%
train_losses = []
val_losses = []

# %%
def validate(model, dataloader, criterion, device):
    """Function to validate the model on validation dataset and return the loss."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for lowres, highres in dataloader:
            outputs = model(lowres.to(device))
            loss = criterion(outputs, highres.to(device))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)  # Average loss

# %%
for epoch in range(num_epochs):
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    print(f"Epoch [{epoch+1}]", end='')
    
    for i, (lowres, highres) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass: Generate SR images
        outputs = model(lowres.to(device))
        
        # Calculate loss (MSE + VGG loss)
        loss = 4 * criterion(outputs, highres.to(device))  # MSE Loss (VGGLoss can be added too)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Convert SR and HR images to numpy for PSNR/SSIM calculation
        sr_img = outputs.cpu().detach().numpy().transpose(0, 2, 3, 1)  # Shape: (batch, H, W, C)
        hr_img = highres.cpu().detach().numpy().transpose(0, 2, 3, 1)  # Shape: (batch, H, W, C)
        
        # Compute PSNR and SSIM for each image in the batch
        batch_psnr = 0
        batch_ssim = 0
        for j in range(sr_img.shape[0]):
            batch_psnr += calculate_psnr(sr_img[j], hr_img[j])
            batch_ssim += calculate_ssim(sr_img[j], hr_img[j])
        
        total_psnr += batch_psnr / sr_img.shape[0]
        total_ssim += batch_ssim / sr_img.shape[0]
        
        if i % 20 == 0:  # Print every 20 mini-batches
            print('[%d, %5d] loss: %.6f, PSNR: %.3f, SSIM: %.3f' % 
                  (epoch + 1, i + 1, running_loss / 10, total_psnr / (i+1), total_ssim / (i+1)))
            running_loss = 0.0
        else:
            print("#", end='')

    # Calculate average training loss for this epoch
    avg_train_loss = running_loss / len(dataloader)
    train_losses.append(avg_train_loss)

    # Validate the model after each epoch
    val_loss = validate(model, test_dataloader, criterion, device)
    val_losses.append(val_loss)
    
    # After each epoch, print average PSNR and SSIM
    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    print(f"Epoch [{epoch+1}] completed with Avg PSNR: {avg_psnr:.3f}, Avg SSIM: {avg_ssim:.3f}, Val Loss: {val_loss:.6f}")

print('Finished Training')

# %%
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# %%
batch = next(iter(dataloader))

# %%

# Get a batch of data from the train set
batch = next(iter(dataloader))

# Extract noisy images from the batch
lowres = batch[0][:5].to(device)  # Select the first 5 images in the batch
highres = batch[1][:5].to(device)  # Corresponding original images

# Generate denoised images using the model
with torch.no_grad():
    srimages = model(lowres)
    srimages = srimages.view(-1, 3, 256, 256)  # Reshape to match the original image size

# Convert torch tensors to numpy arrays
lowres = lowres.cpu().numpy()
highres = highres.cpu().numpy()
srimages = srimages.cpu().numpy()

# Plot the images
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))

for i in range(5):
    # Plot noisy images
    axes[0, i].imshow(np.transpose(lowres[i], (1, 2, 0)))
    axes[0, i].set_title('LowRes Image')
    axes[0, i].axis('off')
    
    # Plot original images
    axes[1, i].imshow(np.transpose(highres[i], (1, 2, 0)))
    axes[1, i].set_title('HiRes Image')
    axes[1, i].axis('off')
    
    # Plot denoised images
    axes[2, i].imshow(np.transpose(srimages[i], (1, 2, 0)))
    axes[2, i].set_title('SR Image')
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()

# %%
with torch.no_grad():
    total_psnr = 0.0
    total_ssim = 0.0
    for i, (lowres, highres) in enumerate(test_dataloader):
        outputs = model(lowres.to(device))
        
        sr_img = outputs.cpu().detach().numpy().transpose(0, 2, 3, 1)
        hr_img = highres.cpu().detach().numpy().transpose(0, 2, 3, 1)
        
        batch_psnr = 0
        batch_ssim = 0
        for j in range(sr_img.shape[0]):
            batch_psnr += calculate_psnr(sr_img[j], hr_img[j])
            batch_ssim += calculate_ssim(sr_img[j], hr_img[j])
        
        total_psnr += batch_psnr / sr_img.shape[0]
        total_ssim += batch_ssim / sr_img.shape[0]
    
    avg_psnr = total_psnr / len(test_dataloader)
    avg_ssim = total_ssim / len(test_dataloader)
    print(f"Test Set Evaluation - Avg PSNR: {avg_psnr:.3f}, Avg SSIM: {avg_ssim:.3f}")


# %%
#saving model weights
weights_dir = "/kaggle/working/"
model_weights_path = os.path.join(weights_dir, 'UNETlatest3.pth')
torch.save(model.state_dict(), model_weights_path)





