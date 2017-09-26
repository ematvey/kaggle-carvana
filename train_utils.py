import datetime
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


def trycuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def scalar(v):
    return float(v.cpu().data.numpy()[0])


def var(x, volatile=False):
    return Variable(trycuda(torch.FloatTensor(x)), volatile=volatile)


CUDA = torch.cuda.is_available()


def tensor_from_numpy(array):
    tensor = torch.from_numpy(array)
    if CUDA:
        tensor = tensor.cuda()
    return tensor


def dice_coef(probs, targets):
    num = targets.size(0)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    return score.mean()


def gpu_preloader_iter(dataloader):
    loader_iter = iter(dataloader)
    bx, by = None, None
    while 1:
        try:
            x, y = bx, by
            bx, by = next(loader_iter)
            if torch.is_tensor(bx):
                bx = bx.cuda(async=True)
            if torch.is_tensor(by):
                by = by.cuda(async=True)
            if x is None or y is None:
                x, y = next(loader_iter)
                if torch.is_tensor(x):
                    x = x.cuda()
                if torch.is_tensor(y):
                    y = y.cuda()
            yield x, y
        except StopIteration:
            if bx is not None:
                yield bx, by
            return


def gpu_iter(dataloader):
    for x, y in dataloader:
        yield x.cuda(), y.cuda()


def run_epoch(epoch, dataset, model, optimizer,
              batch_size=5, step_size=5,
              writer=None, async_loader=True,
              loader_num_workers=0,
              max_grad_norm=0.1):
    start_t = time.time()
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=loader_num_workers if CUDA else 0,
                        pin_memory=CUDA)

    mode = 'TRAIN'
    model = model.train()

    batch_total = 0
    epoch_total = 0
    loss_epoch_sum = 0.0
    loss_batch_sum = 0.0

    optimizer.zero_grad()

    def batch_opt_step():
        nonlocal loss_batch_sum
        nonlocal batch_total

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          max_norm=max_grad_norm,
                                          norm_type=2)

        optimizer.step()
        optimizer.zero_grad()

        batch_total = 0
        loss_batch_sum = 0.0

    if CUDA and async_loader:
        loader_iter = gpu_preloader_iter(loader)
    elif CUDA:
        loader_iter = gpu_iter(loader)
    else:
        loader_iter = iter(loader)

    for x, y in tqdm(loader_iter, total=len(dataset), ncols=0):

        x = Variable(x)
        y = Variable(y)

        out = model(x)
        mask_probs = F.sigmoid(out)
        loss = -dice_coef(mask_probs, y)
        loss.backward()

        loss_value = scalar(loss)
        size = x.size()[0]
        batch_total += size
        epoch_total += size
        loss_batch_sum += loss_value * size
        loss_epoch_sum += loss_value * size

        if not step_size or batch_total >= step_size:
            batch_opt_step()

    if batch_total > 0:
        batch_opt_step()

    took = time.time() - start_t
    if writer:
        writer.add(dict(epoch=epoch, mode=mode,
                        epoch_loss=loss_epoch_sum / epoch_total, epoch_duration_s=took))
    return loss_epoch_sum / epoch_total


def run_eval(epoch, dataset, model, batch_size=5, writer=None, async_loader=True, **kwa):
    t = datetime.datetime.now()
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=True,
                        num_workers=0,
                        pin_memory=torch.cuda.is_available())

    mode = 'EVAL'

    model.eval()

    epoch_total = 0
    score_epoch_sum = 0.0

    if CUDA and async_loader:
        loader_iter = gpu_preloader_iter(loader)
    else:
        loader_iter = iter(loader)

    for i, (x, y) in enumerate(loader_iter):
        size = x.size()[0]
        epoch_total += size
        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)
        out = model(x)
        score = dice_coef((F.sigmoid(out) > 0.5).float(), y)
        score_value = scalar(score)
        if writer:
            writer.add(dict(epoch=epoch, step=i,
                            mode=mode, block_score=score_value))
        score_epoch_sum += score_value * size

    took = str(datetime.datetime.now() - t)
    epoch_score = score_epoch_sum / epoch_total
    if writer:
        writer.add(dict(epoch=epoch, step=i, mode=mode,
                        epoch_score=epoch_score, duration=took))
    return score_epoch_sum / epoch_total


def run_predict(model, submission_fn, batch_size=5, async_loader=True, **kwa):
    from dataset import run_length_encode, CarvanaTestDataset
    from conf import WORKING_SHAPE

    test_dataset = CarvanaTestDataset(WORKING_SHAPE)

    t = datetime.datetime.now()
    loader = DataLoader(test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                        pin_memory=torch.cuda.is_available())

    model.eval()

    epoch_total = 0
    score_epoch_sum = 0.0

    if CUDA and async_loader:
        loader_iter = gpu_preloader_iter(loader)
    else:
        loader_iter = iter(loader)

    with open(submission_fn, 'w') as submission_csv:
        submission_csv.write('img,rle_mask\n')
        for i, (fns, x) in enumerate(tqdm(loader_iter, total=int(len(test_dataset) / batch_size), ncols=0)):
            size = x.size()[0]
            epoch_total += size
            out = model(Variable(x, volatile=True))
            out = out.sigmoid() > 0.5
            oy = out.data.cpu().numpy()
            for fn, mask in zip(fns, oy):
                rle = run_length_encode(oy)
                submission_csv.write('{},{}\n'.format(
                    fn, ' '.join(map(str, rle))))

    took = str(datetime.datetime.now() - t)
    epoch_score = score_epoch_sum / epoch_total
    return score_epoch_sum / epoch_total
