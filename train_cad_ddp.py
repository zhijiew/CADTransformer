import os
import torch
from tqdm import tqdm
from dataset import CADDataLoader, DataLoaderX
from utils.utils_model import create_logger
from config import config, update_config
from models.model import CADTransformer
from utils.utils_model import OffsetLoss
from eval import do_eval, get_eval_criteria
from args import parse_args
from lib.dual_branch_cadformer import DualBranchCADFormer
from loss.align_loss import clip_style_alignment_loss
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1

def main():
    args = parse_args()
    cfg = update_config(config, args)

    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.eval_only:
        logger= create_logger(cfg.log_dir, 'val')
    elif cfg.test_only:
        logger= create_logger(cfg.log_dir, 'test')
    else:
        logger= create_logger(cfg.log_dir, 'train')

    # Distributed Train Config
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
    )
    device = torch.device('cuda:{}'.format(args.local_rank))

    # Create Model
    if args.model == "dual_branch_cadformer":
        model = DualBranchCADFormer(cfg)
    else:
        model = CADTransformer(cfg)
    CE_loss = torch.nn.CrossEntropyLoss().cuda()

    # Create Optimizer
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    model = torch.nn.parallel.DistributedDataParallel(
        module=model.to(device), broadcast_buffers=False,
        device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.train()

    # Load/Resume ckpt
    start_epoch = 0
    if cfg.load_ckpt != '':
        if os.path.exists(cfg.load_ckpt):
            checkpoint = torch.load(cfg.load_ckpt, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.load_ckpt, checkpoint['epoch']))
        else:
            logger.info("=>Failed: no checkpoint found at '{}'".format(cfg.load_ckpt))
            exit(0)

    if cfg.resume_ckpt != '':
        if os.path.exists(cfg.load_ckpt):
            checkpoint = torch.load(cfg.resume_ckpt, map_location=torch.device("cpu"))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            logger.info(f'=> resume checkpoint: {cfg.resume_ckpt} (epoch: {epoch})')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            logger.info("=>Failed: no checkpoint found at '{}'".format(cfg.resume_ckpt))
            exit(0)
    # Set up Dataloader
    torch.multiprocessing.set_start_method('spawn', force=True)
    val_dataset = CADDataLoader(split='val', do_norm=cfg.do_norm, cfg=cfg)
    val_dataloader = DataLoaderX(args.local_rank, dataset=val_dataset,
                                batch_size=cfg.test_batch_size, shuffle=False,
                                num_workers=cfg.WORKERS, drop_last=False)
    # Eval Only
    if args.local_rank == 0:
        if cfg.eval_only:
            eval_F1 = do_eval(model, val_dataloader, logger, cfg)
            exit(0)

    test_dataset = CADDataLoader(split='test', do_norm=cfg.do_norm, cfg=cfg)
    test_dataloader = DataLoaderX(args.local_rank, dataset=test_dataset,
                                 batch_size=cfg.test_batch_size, shuffle=False,
                                 num_workers=cfg.WORKERS, drop_last=False)
    # Test Only
    if args.local_rank == 0:
        if cfg.test_only:
            eval_F1 = do_eval(model, test_dataloader, logger, cfg)
            exit(0)

    train_dataset = CADDataLoader(split='train', do_norm=cfg.do_norm, cfg=cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoaderX(args.local_rank, dataset=train_dataset,
                                  sampler=train_sampler, batch_size=cfg.batch_size,
                                  num_workers=cfg.WORKERS, drop_last=True)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    best_F1, eval_F1 = 0, 0
    best_epoch = 0
    global_epoch = 0

    print("> start epoch", start_epoch)
    for epoch in range(start_epoch, cfg.epoch):
        logger.info(f"=> {cfg.log_dir}")

        logger.info("\n\n")
        logger.info(f'Epoch {global_epoch + 1} ({epoch + 1}/{cfg.epoch})')
        lr = max(cfg.learning_rate * (cfg.lr_decay ** (epoch // cfg.step_size)),
                 cfg.LEARNING_RATE_CLIP)
        if epoch <= cfg.epoch_warmup:
            lr = cfg.learning_rate_warmup

        logger.info(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = cfg.MOMENTUM_ORIGINAL * (cfg.MOMENTUM_DECCAY ** (epoch // cfg.step_size))
        if momentum < 0.01:
            momentum = 0.01
        logger.info(f'BN momentum updated to: {momentum}')
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        model = model.train()

        # training loops
        with tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9) as _tqdm:
            for i, batch in enumerate(_tqdm):
                optimizer.zero_grad()

                if args.model == "dual_branch_cadformer":
                    if isinstance(batch, dict):
                        image = batch.get("img") or batch.get("image")
                        mask = batch.get("mask")
                        vec_x = batch.get("vec_x")
                        vec_edge_index = batch.get("vec_edge_index")
                        vec_label = batch.get("vec_label")
                    else:
                        image = batch[0]
                        mask = batch[2] if len(batch) > 2 else None
                        vec_x, vec_edge_index, vec_label = None, None, None

                    outputs = model(image, vec_x=vec_x, vec_edge_index=vec_edge_index)
                    raster_logits = outputs["raster_logits"]
                    vector_logits = outputs.get("vector_logits", [])
                    rast_z = outputs.get("rast_z")
                    vect_z = outputs.get("vect_z")

                    if mask is not None:
                        loss_raster = F.cross_entropy(raster_logits, mask)
                    else:
                        loss_raster = torch.tensor(0.0, device=raster_logits.device)

                    vector_losses = []
                    if vec_label is not None and vector_logits:
                        for logit_item, label_item in zip(vector_logits, vec_label):
                            vector_losses.append(F.cross_entropy(logit_item, label_item.to(logit_item.device)))
                    if vector_losses:
                        loss_vector = torch.stack(vector_losses).mean()
                    else:
                        loss_vector = torch.tensor(0.0, device=raster_logits.device)

                    if vect_z is not None and rast_z is not None:
                        loss_align = clip_style_alignment_loss(rast_z, vect_z, temperature=cfg.align_temperature)
                    else:
                        loss_align = torch.tensor(0.0, device=raster_logits.device)

                    loss = cfg.lambda_raster * loss_raster + cfg.lambda_vector * loss_vector + cfg.lambda_align * loss_align
                    loss_seg = loss_raster
                else:
                    image, xy, target, rgb_info, nns, offset_gt, inst_gt, index, basename = batch
                    seg_pred = model(image, xy, rgb_info, nns)
                    seg_pred = seg_pred.contiguous().view(-1, cfg.num_class+1)
                    target = target.view(-1, 1)[:, 0]

                    loss_seg = CE_loss(seg_pred, target)
                    loss = loss_seg

                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(loss=loss.item(), l_seg=loss_seg.item())

                if i % args.log_step == 0 and args.local_rank == 0:
                    logger.info(f'Train loss: {round(loss.item(), 5)}, loss seg: {round(loss_seg.item(), 5)})')

        # Save last
        if args.local_rank == 0:
            logger.info('Save last model...')
            savepath = os.path.join(cfg.log_dir, 'last_model.pth')
            state = {
                'epoch': epoch,
                'best_F1': best_F1,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        # assert validation?
        eval = get_eval_criteria(epoch)

        if args.local_rank == 0:
            if eval:
                logger.info('> do validation')
                eval_F1 = do_eval(model, val_dataloader, logger, cfg)
        # Save ckpt
        if args.local_rank == 0:
            if eval_F1 > best_F1:
                best_F1 = eval_F1
                best_epoch = epoch
                logger.info(f'Save model... Best F1:{best_F1}, Best Epoch:{best_epoch}')
                savepath = os.path.join(cfg.log_dir, 'best_model.pth')
                state = {
                    'epoch': epoch,
                    'best_F1': best_F1,
                    'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1


if __name__ == '__main__':
    main()
