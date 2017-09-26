from dataset import CarvanaDataset, CarvanaTestDataset, TRAIN_INDEX, DEV_INDEX
from models import MyUNet

build_prod_model = MyUNet
build_debug_model = MyUNet

W, H = 1920, 1280
WORKING_SHAPE = (H, W)

train_dataset = CarvanaDataset(TRAIN_INDEX, WORKING_SHAPE)
dev_dataset = CarvanaDataset(DEV_INDEX, WORKING_SHAPE)
test_dataset = CarvanaTestDataset(WORKING_SHAPE)

experiment = 'unet'
hyperparams = dict(
    lr=1e-3,
    max_epochs=3000,
    weight_decay=1e-6,
    batch_size=1,
    grad_step_size=5,
    loader_num_workers=3,
    pin_memory=True,
)
