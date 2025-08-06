from facenet_pytorch import MTCNN, InceptionResnetV1

from PIL import Image
import pdb

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=0, device='cuda:6')


img = Image.open("/data/bohong/object_L/data/examples/scenery.png")
img_cropped = mtcnn(img, save_path="/data/bohong/object_L/data/examples/scenery_cropped.png")
pdb.set_trace()
# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))