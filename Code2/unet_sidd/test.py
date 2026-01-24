model.load_state_dict(torch.load("unet_sidd.pth"))
model.eval()

# run full image through sliding window or padding
# compute PSNR / save output images
