import sys
sys.path.append('../../repos/pips2')
import os
import json

import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
import utils.misc
import random
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
from torch.utils.data import Dataset, DataLoader
# from datasets.tapviddataset_fullseq import TapVidDavis
from mydataset import TapVidDavis, TapVidDepthDavis

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=[512,896], nargs='+')
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--log_freq", type=int, default=99)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--mode", type=str, default="depth_tapvid_davis_first")
    parser.add_argument("--data_root", type=str, default="./tap/sketch_tapvid_davis")
    parser.add_argument("--proportions", type=float, default=[0.0, 0.0, 0.0], nargs='+')
    parser.add_argument("--init_dir", type=str, default='./reference_model')
    parser.add_argument("--device_ids", type=list, default=[0])
    parser.add_argument("--n_pool", type=int, default=1000)
    return parser.parse_args()

def create_pools(n_pool=1000):
    pools = {}
    pool_names = [
        'd_1',
        'd_2',
        'd_4',
        'd_8',
        'd_16',
        'd_avg',
        'median_l2',
        'survival',
        'ate_all',
        'ate_vis',
        'ate_occ',
        'total_loss',
    ]
    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version='np')
    return pools

def test_on_fullseq(model, d, sw, iters=8, S_max=8, image_size=(384,512)):
    metrics = {}

    rgbs = d['rgbs'].cuda().float() # B,S,C,H,W
    trajs_g = d['trajs'].cuda().float() # B,S,N,2
    valids = d['valids'].cuda().float() # B,S,N

    B, S, C, H, W = rgbs.shape
    B, S, N, D = trajs_g.shape
    assert(D==2)
    assert(B==1)
    # print('this video is %d frames long' % S)

    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = image_size
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    rgbs = rgbs_.reshape(B, S, 3, H_, W_)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    H, W = H_, W_
    
    # zero-vel init
    trajs_e = trajs_g[:,0].repeat(1,S,1,1)
    
    cur_frame = 0
    done = False
    feat_init = None
    while not done:
        end_frame = cur_frame + S_max

        if end_frame > S:
            diff = end_frame-S
            end_frame = end_frame-diff
            cur_frame = max(cur_frame-diff,0)
        # print('working on subseq %d:%d' % (cur_frame, end_frame))

        traj_seq = trajs_e[:, cur_frame:end_frame]
        rgb_seq = rgbs[:, cur_frame:end_frame]
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]
            
        preds, preds_anim, feat_init, _ = model(traj_seq, rgb_seq, iters=iters, feat_init=feat_init)
        # print('preds', preds[-1].shape, 'preds_anim', preds_anim[-1].shape)
        # print('preds', preds[-1].min().item(), preds[-1].max().item(), 'preds_anim', preds_anim[-1].min().item(), preds_anim[-1].max().item())

        trajs_e[:, cur_frame:end_frame] = preds[-1][:, :S_local]
        trajs_e[:, end_frame:] = trajs_e[:, end_frame-1:end_frame] # update the future with new zero-vel

        if sw is not None and sw.save_this:
            traj_seq_e = preds[-1]
            traj_seq_g = trajs_g[:,cur_frame:end_frame]
            valid_seq = valids[:,cur_frame:end_frame]
            prep_rgbs = utils.improc.preprocess_color(rgb_seq)
            gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
            gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', traj_seq_g, gray_rgbs[0:1].mean(dim=1), valids=valid_seq, cmap='winter', only_return=True))
            rgb_vis = []
            for tre in preds_anim:
                ate = torch.norm(tre - traj_seq_g, dim=-1) # B,S,N
                ate_all = utils.basic.reduce_masked_mean(ate, valid_seq, dim=[1,2]) # B
                rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valid_seq, only_return=True, cmap='spring', frame_id=ate_all[0]))
            print('rgb_vis', rgb_vis[0].shape, rgb_vis[0].min().item(), rgb_vis[0].max().item())
            sw.summ_rgbs('3_test/animated_trajs_on_rgb_cur%02d' % cur_frame, rgb_vis)
        
        if end_frame >= S:
            done = True
        else:
            cur_frame = cur_frame + S_max - 1


    d_sum = 0.0
    thrs = [1,2,4,8,16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics['d_avg'] = d_avg
    
    sur_thr = 16
    dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0

    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True)
    metrics['median_l2'] = median_l2.mean().item()

    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgbs[0:1,0], valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
        sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=d_avg)
        st = 4
        sw.summ_traj2ds_on_rgbs2('2_outputs/trajs_e_on_rgbs2', trajs_e[0:1,::st], valids[0:1,::st], prep_rgbs[0:1,::st], valids=valids[0:1,::st], frame_ids=list(range(0,S,st)))

    return metrics

def eval_cycle(model, model_name, mode, data_root, video_path, proportions, image_size, shuffle, n_pool, log_freq, max_iters, iters, S, log_dir):
    writer_x = SummaryWriter(log_dir + '/' + model_name + '/x', max_queue=10, flush_secs=60)
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    sys.path.append("../")
    from data_utils import get_sketch_data_path, get_depth_root_from_data_root, get_perturbed_data_path
    exp_type, set_type = mode.split('_')[0], '_'.join(mode.split('_')[1:])
    
    if exp_type == 'realworld':
        from mydataset import RealWorldDataset
        dataset_x = RealWorldDataset(data_root, proportions, image_size)
    else:
        if exp_type == 'sketch':
            PATHS = get_sketch_data_path(data_root)
        elif exp_type == 'perturbed':
            PATHS = get_perturbed_data_path(data_root)
            
        dataset_type, dataset_root, queried_first = PATHS[set_type]
        print('loading %s dataset...' % dataset_type, dataset_root, "proportions", proportions, "image_size", image_size)
        
        dataset_x = TapVidDepthDavis(
            dataset_type=dataset_type,
            data_root=dataset_root,
            depth_root=get_depth_root_from_data_root(dataset_root) \
                if exp_type == 'sketch' else os.path.join(video_path, "video_depth_anything"),
            proportions=proportions,
            queried_first=queried_first,
            image_size=image_size,
        )
                
    dataloader_x = DataLoader(
        dataset_x,
        batch_size=1,
        shuffle=shuffle,
        num_workers=1)
    iterloader_x = iter(dataloader_x)    
    
    pools_x = create_pools(n_pool)

    global_step = 0
    max_iters = min(max_iters, len(dataset_x))
    while global_step < max_iters:
        global_step += 1
        iter_start_time = time.time()
        with torch.no_grad():
            torch.cuda.empty_cache()
        sw_x = utils.improc.Summ_writer(
            writer=writer_x,
            global_step=global_step,
            log_freq=log_freq,
            fps=min(S,8),
            scalar_freq=1,
            just_gif=True)
        try:
            sample = next(iterloader_x)
        except StopIteration:
            iterloader_x = iter(dataloader_x)
            sample = next(iterloader_x)
        if sample['trajs'].shape[2] == 0:
            print("encounter 0 points sample")
            continue
        iter_rtime = time.time()-iter_start_time
        with torch.no_grad():
            metrics = test_on_fullseq(model, sample, sw_x, iters=iters, S_max=S, image_size=image_size)
        for key in list(pools_x.keys()):
            if key in metrics:
                pools_x[key].update([metrics[key]])
                sw_x.summ_scalar('_/%s' % (key), pools_x[key].mean())
        iter_itime = time.time()-iter_start_time
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; d_x %.1f; sur_x %.1f; med_x %.1f' % (
            model_name, global_step, max_iters, iter_rtime, iter_itime,
            pools_x['d_avg'].mean(), pools_x['survival'].mean(), pools_x['median_l2'].mean()))
        
    # with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
    #     json.dump({'d_avg': pools_x['d_avg'].mean(), 'survival': pools_x['survival'].mean(), 'median_l2': pools_x['median_l2'].mean()}, f)
            
    writer_x.close()
    return {
        "average_pts_within_thresh": pools_x['d_avg'].mean()
    }
    
def main(
        B=1, # batchsize 
        S=128, # seqlen
        stride=8, # spatial stride of the model 
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        shuffle=False, # dataset shuffling
        log_freq=99, # how often to make image summaries
        max_iters=30, # how many samples to test
        log_dir='./logs_test_on_tap',
        mode="depth_tapvid_davis_first",
        data_root="./datasets/tap/sketch_tapvid_rgbs",
        proportions=[0.0, 0.0, 0.0],
        init_dir='./reference_model',
        device_ids=[0], 
        n_pool=1000, # how long the running averages should be
):
    device = 'cuda:%d' % device_ids[0]

    # the idea in this file is:
    # load a ckpt, and test it in tapvid,
    # tracking points from frame0 to the end.
    
    exp_name = 'tap00' # copy from dev repo
    exp_name = 'tap01' # clean up

    assert(B==1) # B>1 not implemented here
    assert(image_size[0] % 32 == 0)
    assert(image_size[1] % 32 == 0)
    
    # autogen a descriptive name
    model_name = "%d_%d" % (B,S)
    model_name += "_i%d" % (iters)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    exp_type, set_type = mode.split('_')[0], '_'.join(mode.split('_')[1:])

    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    utils.misc.count_parameters(model)

    _ = saverloader.load(init_dir, model.module)
    model.eval()

    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, f"evaluation_results.txt")
    
    if exp_type == 'sketch' or exp_type == 'realworld':
        scores = eval_cycle(model, model_name, mode, data_root, data_root, proportions, image_size, shuffle, n_pool, log_freq, max_iters, iters, S, log_dir)

        with open(output_file, "w") as f:
            for key, score in scores.items():
                f.write(f"{key}: {score}\n")
    
    elif exp_type == 'perturbed':
        total = {}
        pert_sev_results = {}  # New dictionary for storing perturbation-severity pairs
        pert_root = os.path.join(data_root, "perturbations")
        for perturbation in os.listdir(pert_root):
            pert_path = os.path.join(pert_root, perturbation)
            
            for severity in range(1, 6, 2):  # Loop through severity levels
                sev_path = os.path.join(pert_path, f"severity_{severity}")
                print(sev_path)

                # Evaluate for current perturbation-severity pair
                score = eval_cycle(model, model_name, mode, data_root, sev_path, proportions, image_size, shuffle, n_pool, log_freq, max_iters, iters, S, log_dir)

                # Store results for perturbation-severity pair
                key = f"{perturbation}-severity_{severity}"
                pert_sev_results[key] = {
                    'average_pts_within_thresh': score['average_pts_within_thresh']
                }

                # print(f"Processed {key}")

                # Aggregate per perturbation
                total.setdefault(perturbation, []).append(score['average_pts_within_thresh'])

        # Compute final per-perturbation averages
        perturbation_avg = {
            perturbation: {
                'average_pts_within_thresh': np.mean(total[perturbation])
            }
            for perturbation in total
        }

        # Compute final overall averages
        results = {
            'average_pts_within_thresh': np.mean(list(total.values()))
        }

        # Save results to a file
        with open(output_file, "w") as f:
            # Summary of all perturbations
            f.write("Summary of all perturbations\n")
            for metric, scores in results.items():
                f.write(f"all-{metric}: {scores}\n")
            f.write("\n")
            
            # Summary of all perturbation-severity pairs
            f.write("Summary of all perturbation-severity pairs\n")
            for perturbation in perturbation_avg.keys():
                # f.write(f"{perturbation}\n")
                for metric, score in perturbation_avg[perturbation].items():
                    f.write(f"{perturbation}-{metric}: {score}\n")
            f.write("\n")
                    
            # Write perturbation-severity pair results
            f.write("Results for each perturbation-severity pair\n")
            for each_perturbation in pert_sev_results.keys():
                for metric, score in pert_sev_results[each_perturbation].items():
                    f.write(f"{each_perturbation}-{metric}: {score}\n")
            f.write("\n")  
            
if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))