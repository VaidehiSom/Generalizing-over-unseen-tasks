from torchvision import transforms

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'
MODEL_FILE = 'model.pt'

# available actions # TODO: Continuos space- joint velocities
available_actions = [[0, 0, 0],  # no action
                     [1, 0, 0],  # up
                     [-1, 0, 0],  # down
                     [0, 1, 0],  # right
                     [0, -1, 0],  # left
                     [0, 0, 1],  # open gripper
                     [0, 0, -1] ]  # close gripper

# transformations for training/testing
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(1),
    transforms.Pad((12, 12, 12, 0)),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])