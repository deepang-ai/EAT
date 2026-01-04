import os
import sys
from datetime import datetime
from typing import Dict

import monai
# import pytz
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model
import warnings

warnings.filterwarnings('ignore')


def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                    post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # Training
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch[0])
        total_loss = 0
        log = ''
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch[1])
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += alpth * loss
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss:1.5f} {log}',
            flush=True)
        step += 1
        # break
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
            f'Train/mean {metric_name}': float(batch_acc.mean())})

    accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}')
    accelerator.log(metric, step=epoch)
    return step


@torch.no_grad()
def val_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int):
    # Validation
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch[0], model)
        total_loss = 0
        log = ''
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch[1])
            accelerator.log({'Val/' + name: float(loss)}, step=step)
            log += f' {name} {float(loss):1.5f} '
            total_loss += loss
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])
        accelerator.log({
            'Val/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}',
            flush=True)
        step += 1
    metric = {}
    if config.trainer.dataset_choose != 'EDD_seg':
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
            metrics[metric_name].reset()
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean())})

        accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}')
        accelerator.log(metric, step=epoch)
    else:
        for metric_name in metrics:
            batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
            metrics[metric_name].reset()
            if batch_acc.size() == torch.Size([]) or batch_acc.size() == torch.Size([1]):
                metric.update({
                    f'Val/mean {metric_name}': float(batch_acc.mean()),
                })
            else:
                metric.update({
                    f'Val/mean {metric_name}': float(batch_acc.mean()),
                    f'Val/BE {metric_name}': float(batch_acc[0]),
                    f'Val/cancer {metric_name}': float(batch_acc[1]),
                    f'Val/HGD {metric_name}': float(batch_acc[2]),
                    f'Val/polyp {metric_name}': float(batch_acc[3]),
                    f'Val/suspicious {metric_name}': float(batch_acc[4])})
    return torch.Tensor([metric['Val/mean dice_metric']]).to(accelerator.device), metric, step


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(328)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print('Load Model...')
    model = give_model(config)

    image_size = config.dataset[config.trainer.dataset_choose].image_size

    accelerator.print('Load Dataloader...')
    if config.trainer.dataset_choose == 'CVC_ClinicDB':
        from src.CVCLoader import get_dataloader

        train_loader, val_loader = get_dataloader(config, dataset_choose='CVC_ClinicDB')
        include_background = False
    elif config.trainer.dataset_choose == 'Kvasir_SEG':
        from src.CVCLoader import get_dataloader

        train_loader, val_loader = get_dataloader(config, dataset_choose='Kvasir_SEG')
        include_background = False
    elif config.trainer.dataset_choose == 'EDD_seg':
        from src.EDDLoader import get_dataloader

        train_loader, val_loader = get_dataloader(config)
        include_background = True
    elif config.trainer.dataset_choose == 'PolypGen':
        from src.PolypGenLoder import get_dataloader

        train_loader, val_loader = get_dataloader(config)
        include_background = True
    elif config.trainer.dataset_choose == 'Sun_seg':
        from src.SunsegLoader import get_dataloader

        train_loader, val_loader = get_dataloader(config)
        include_background = True

    inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 2), overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=include_background,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        'miou_metric': monai.metrics.MeanIoU(include_background=include_background),
        'f1': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name='f1 score'),
        'precision': monai.metrics.ConfusionMatrixMetric(include_background=include_background,
                                                         metric_name="precision"),
        'recall': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="recall"),
        'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=include_background,
                                                             reduction=monai.utils.MetricReduction.MEAN_BATCH,
                                                             get_not_nans=True)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    # Define training parameters
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }

    step = 0
    best_eopch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = torch.tensor(0)
    best_class = []
    # # Try to load pre-trained model
    # model = load_pretrain_model(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin", model,
    #                             accelerator)

    # Try to resume training
    if config.trainer.resume:
        model, optimizer, scheduler, starting_epoch, train_step, best_acc, best_class = utils.resume_train_state(model,
                                                                                                                 '{}'.format(
                                                                                                                     config.finetune.checkpoint),
                                                                                                                 optimizer,
                                                                                                                 scheduler,
                                                                                                                 train_loader,
                                                                                                                 accelerator)
        val_step = train_step

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader, val_loader)
    best_acc = best_acc.to(accelerator.device)

    # Start training
    accelerator.print("Start Training!")

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # Training
        step = train_one_epoch(model, loss_functions, train_loader,
                               optimizer, scheduler, metrics,
                               post_trans, accelerator, epoch, step)
        # Validation
        mean_acc, batch_acc, val_step = val_one_epoch(model, loss_functions, inference, val_loader,
                                                      config, metrics, val_step,
                                                      post_trans, accelerator, epoch)

        # Save model
        if mean_acc > best_acc:
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best")
            best_acc = mean_acc
            best_class = batch_acc
            best_eopch = epoch
        accelerator.print('Cheakpoint...')
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint")
        torch.save({'epoch': epoch, 'best_acc': best_acc, 'best_class': batch_acc},
                   f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint/epoch.pth.tar')

        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] best acc:{best_acc}, Now : mean acc: {mean_acc}, mean class: {batch_acc}')

    accelerator.print(f"best acc: {best_acc}")
    accelerator.print(f"best class : {best_class}")
    accelerator.print(f"best epochs: {best_eopch}")
    sys.exit(1)
