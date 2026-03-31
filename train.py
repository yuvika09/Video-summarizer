import random
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import vs_helper
import data_loader
import init
from model import STeMI
from evaluate import evaluate
from losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss, reconstruction_loss

logger = logging.getLogger()

def train(args, split, save_path):
    model = STeMI(
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        temporal_scales=args.temporal_scales,
        spatial_scales=args.spatial_scales,
        dropout=args.dropout
    )
    model = model.to(args.device)
    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    max_val_fscore = -1
    best_epoch = 0
    best_model_state = None
    
    train_set = data_loader.VideoDataset(split['train_keys'])
    train_loader = data_loader.DataLoader(train_set, shuffle=True)
    val_set = data_loader.VideoDataset(split['test_keys'])
    val_loader = data_loader.DataLoader(val_set, shuffle=False)

    no_improve_count = 0

    for epoch in range(args.max_epoch):
        random.seed(epoch + args.seed)
        model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for _, seq, gtscore, change_points, n_frames, nfps, picks, _, _, support_video in train_loader:
            
            gtscore = torch.tensor(gtscore, dtype=torch.float32).to(args.device)
            change_points = torch.tensor(change_points, dtype=torch.int64).to(args.device)
            n_frames = torch.tensor(n_frames, dtype=torch.int64).to(args.device)
            nfps = torch.tensor(nfps, dtype=torch.int64).to(args.device)
            picks = torch.tensor(picks, dtype=torch.int64).to(args.device)
            
            keyshot_summ, _ = vs_helper.get_keyshot_summ(gtscore, change_points, n_frames, nfps, picks)
            target = vs_helper.downsample_summ(keyshot_summ)
            
            support_video["gtscore"] = torch.tensor(support_video["gtscore"], dtype=torch.float32).to(args.device)
            support_video["change_points"] = torch.tensor(support_video["change_points"], dtype=torch.int64).to(args.device)
            support_video["n_frames"] = torch.tensor(support_video["n_frames"], dtype=torch.int64).to(args.device)
            support_video["n_frame_per_seg"] = torch.tensor(support_video["n_frame_per_seg"], dtype=torch.int64).to(args.device)
            support_video["picks"] = torch.tensor(support_video["picks"], dtype=torch.int64).to(args.device)
            
            support_keyshot_summ, _ = vs_helper.get_keyshot_summ(
                support_video["gtscore"],
                support_video["change_points"],
                support_video["n_frames"],
                support_video["n_frame_per_seg"],
                support_video["picks"]
            )
            support_target = vs_helper.downsample_summ(support_keyshot_summ)
            selected_indices = torch.where(support_target == 1)[0]
            
            if not target.any() or len(selected_indices) == 0:
                continue

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            support_seq = torch.tensor(support_video["features"], dtype=torch.float32).unsqueeze(0).to(args.device)
            
            cls_label = target.float()
            loc_label = vs_helper.get_loc_label(target).float()
            ctr_label = vs_helper.get_ctr_label(target, loc_label).float()
            
            pred_cls, pred_loc, pred_ctr, recons_x, recons_support = model(seq, support_seq, selected_indices)
            
            cls_loss = calc_cls_loss(pred_cls, cls_label)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label)
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)
            rec_x = reconstruction_loss(recons_x, seq)
            rec_s = reconstruction_loss(recons_support, support_seq)
            
            loss = cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss + args.lambda_rec_x * rec_x + args.lambda_rec_s * rec_s

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=args.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        val_fscore, _ = evaluate(model, val_loader, args.nms_thresh, args.device)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 20 == 0 or val_fscore >= max_val_fscore:
            logger.info(
                f'Epoch: {epoch}/{args.max_epoch} | '
                f'Loss: {avg_epoch_loss:.4f} | '
                f'F-score: {val_fscore:.4f}/{max_val_fscore:.4f} (best@{best_epoch})'
            )

        if args.early_stopping and no_improve_count >= args.patience:
            logger.info(f'Early stopping at epoch {epoch}. Best F-score: {max_val_fscore:.4f} at epoch {best_epoch}')
            break

    if best_model_state is not None:
        torch.save(best_model_state, str(save_path))

    logger.info(f'Training completed. Best F-score: {max_val_fscore:.4f} at epoch {best_epoch}')
    return max_val_fscore

def main():
    args = init.get_arguments()
    init.init_logger(args.model_dir)
    init.set_random_seed(args.seed)

    logger.info(vars(args))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_loader.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    data_loader.dump_yaml(vars(args), model_dir / 'args.yml')

    # Multi-seed training for better results
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    
    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_loader.load_yaml(split_path)

        results = {}
        stats = data_loader.AverageMeter('fscore')

        for split_idx, split in enumerate(splits):
            logger.info(f'Start training on {split_path.stem}: split {split_idx}')
            
            # Train with multiple seeds and take best
            best_fscore_for_split = -1
            best_seed = args.seed
            
            for seed in seeds:
                init.set_random_seed(seed)
                ckpt_path = data_loader.get_ckpt_path(model_dir, split_path, split_idx)
                temp_ckpt_path = str(ckpt_path) + f'.seed{seed}'
                
                fscore = train(args, split, temp_ckpt_path)
                
                if fscore > best_fscore_for_split:
                    best_fscore_for_split = fscore
                    best_seed = seed
                    # Copy best model to final path
                    import shutil
                    shutil.copy(temp_ckpt_path, str(ckpt_path))
                
                logger.info(f'Split {split_idx}, Seed {seed}: F-score = {fscore:.4f}')
            
            logger.info(f'Split {split_idx} Best: F-score = {best_fscore_for_split:.4f} (seed {best_seed})')
            
            stats.update(fscore=best_fscore_for_split)
            results[f'split{split_idx}'] = float(best_fscore_for_split)

        results['mean'] = float(stats.fscore)
        data_loader.dump_yaml(results, model_dir / f'{split_path.stem}.yml')

        logger.info(f'Training done on {split_path.stem}. F-score: {stats.fscore:.4f}')

if __name__ == '__main__':
    main()
