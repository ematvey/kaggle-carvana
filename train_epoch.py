import sys
import torch
from conf import build_prod_model, hyperparams as hpr, experiment, train_dataset, dev_dataset
from img_utils import output_some_images
from train_utils import trycuda, run_epoch
import argparse

parser = argparse.ArgumentParser('train_epoch.py')
parser.add_argument('epoch', type=int)

args = parser.parse_args()
epoch = args.epoch

print('epoch {} with {}'.format(epoch, hpr))

model = build_prod_model()

checkpoint_fn = 'checkpoint.torch'


def save():
    optimizer.zero_grad()
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, checkpoint_fn)


optimizer = torch.optim.Adam(
    model.parameters(), lr=hpr['lr'], weight_decay=hpr['weight_decay'])

model = trycuda(model)

try:
    state = torch.load(checkpoint_fn)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
except FileNotFoundError:
    pass

try:
    train_loss = run_epoch(
        epoch, train_dataset, model, optimizer,
        batch_size=hpr['batch_size'],
        step_size=hpr['grad_step_size'],
        loader_num_workers=hpr['loader_num_workers'],
        async_loader=True,
    )
    print('epoch {}, train loss: {}'.format(epoch, train_loss))
    save()
    output_some_images(model, train_dataset)
    dev_dice = run_eval(epoch, dev_dataset, model, batch_size=n)
    print('epoch {}, dev loss: {}'.format(epoch, dev_loss))

except KeyboardInterrupt:
    import IPython
    IPython.embed(banner1='\nkeyboard interrupt\n')
    sys.exit(1)
