from PIL import Image
import  numpy as np

black = Image.open("random/48/train_GT/00027.png")
black_np = np.asarray(black)
print(np.mean(black_np[:,:,:]))