import torch

from conf import build_prod_model, hyperparams, experiment
from train_utils import trycuda, run_predict
from writer import DefaultWriter

epoch = 1
n = hyperparams['batch_size']

model = build_prod_model()

checkpoint_fn = 'checkpoint.torch'

writer = DefaultWriter(experiment + '.log')

model = trycuda(model)
state = torch.load(checkpoint_fn)
model.load_state_dict(state['model'])

print(model)
print('starting make_submission')

run_predict(model, 'submission.csv', batch_size=n*2, writer=writer)
