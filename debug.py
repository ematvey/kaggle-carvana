import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

import tools
from conf import WORKING_SHAPE, build_debug_model
from dataset import CarvanaDataset, TRAIN_INDEX
from train_utils import var, trycuda

dataset = CarvanaDataset(TRAIN_INDEX, WORKING_SHAPE)
loader = DataLoader(dataset,
                    batch_size=1,
                    sampler=SequentialSampler(dataset),
                    drop_last=False,
                    pin_memory=torch.cuda.is_available())

eval = False

x, mask = next(iter(loader))
x = var(x, volatile=eval)
mask = var(mask, volatile=eval)

model = build_debug_model(verbose=1)
if not eval:
    model.train()
else:
    model.eval()
model = trycuda(model)

print(tools.torch_summarize(model))

out = model(x)

print(out.size())
print('model size, mb', tools.params_mem_size_mb(model))
print('graph size, mb', tools.graph_mem_size_mb(out))
