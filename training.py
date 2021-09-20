'''Implements a generic training loop.
'''

import torch
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import os
import shutil
import loss_functions, utils
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, data_shape, loss_fn, loss_schedules=None, weight=1):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    epoch_start = 0
    train_losses = []
    
    if os.path.exists(model_dir+'/checkpoints'):
        val = input("The model directory %s exists. Load latest run? (y/n)" % model_dir)
        if val == 'y':
            filename = utils.find_latest_checkpoint(model_dir)
            model, optim, epoch_start, train_losses =  utils.load_checkpoint(model, optim, filename)
            

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    for epoch in range(epoch_start, epochs):
        if not epoch % epochs_til_checkpoint and epoch:
            print('epoch:', epoch )
        
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': train_losses,
                        },  os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
        
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                       np.array(train_losses))
            
       
        for step, (model_input, gt) in enumerate(train_dataloader):
            start_time = time.time()

            model_input = {key: value.cuda() for key, value in model_input.items()}
            train_loss = 0.

            # start minibatching
            minibatch_size = int(74/2) # number of slices to send at a time, must be integer multuple of data_shape[0] = 74

            # flattened detector with shape (1, 74*77*394, 3)
            slice_size = data_shape[1]*data_shape[2]
            num_batches = int( model_input['coords'].size(1) / (minibatch_size*slice_size))
            for mini in range(num_batches):

                batch_model_in = model_input['coords'][:, slice_size*mini*minibatch_size:slice_size*(mini+1)*minibatch_size, :]
                batch_gt = gt['coords'][:, :, slice_size*mini*minibatch_size:slice_size*(mini+1)*minibatch_size, :]

                model_output = model(batch_model_in)

                losses = loss_fn(model_output, batch_gt)

                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

            train_loss = torch.sum(train_loss)
            
        if not total_steps % steps_til_summary:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': train_losses,
                    },  os.path.join(checkpoints_dir, 'model_current.pth'))

        optim.zero_grad()
        train_loss.retain_grad()
        train_loss.backward(retain_graph=True)
        optim.step()

        total_steps += 1


    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir, 'model_final.pth'))
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
               np.array(train_losses))
    
    # TODO: migrate to utils file
    #Plot and save loss
    x_steps = np.linspace(0, len(train_losses), num=len(train_losses))

    plt.figure(tight_layout=True)
    plt.plot(x_steps, train_losses)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt_name = os.path.join(model_dir, 'total_loss.png')
    plt.savefig(plt_name, dpi=300, bbox_inches='tight', facecolor='w')
    plt.clf()
    

    # Make images                       
    model_output = model(model_input['coords'])
    ground_truth_video = np.reshape(
          gt['coords'].cpu().detach().numpy(), 
          (data_shape[0], data_shape[1], data_shape[2], -1)
        )
    ground_truth_video = ground_truth_video[:,:,:,0]
    
    predict_video = np.reshape(
            model_output['model_out'].cpu().detach().numpy(), 
            (data_shape[0], data_shape[1], data_shape[2], -1)
        )
    predict_video = predict_video[:,:,:,0]
    
    # Plot pred plots
    log_gt = []
    log_pred = []
    d = []
    st = []
    
    for step in range(predict_video.shape[0]):
    #Plot y pred vs gt
        pred = (predict_video[step]-predict_video[step].min())/(predict_video[step].max()-predict_video[step].min())
        plt.figure(tight_layout=True)
        plt.scatter(-np.log(ground_truth_video[step]+1e-7), -np.log(predict_video[step]+1e-7))        
        plt.xlabel('Truth')
        plt.ylabel('Pred')
        plt.xlim(-1,17)
        plt.ylim(-1,17)
        plt.grid(True)
        plt_name = os.path.join(model_dir, '{:05d}_pred_vs_truth.png'.format(step))
        plt.savefig(plt_name, dpi=300, bbox_inches='tight', facecolor='w')
        plt.clf()
        
        log_gt.append(-np.log(ground_truth_video[step]+1e-7))
        log_pred.append(-np.log(predict_video[step]+1e-7))
        
        # Histogram of value overlaps
        plt.hist(pred.flatten(), alpha=0.5, label='predicted')
        plt.hist(ground_truth_video[step].flatten(), alpha=0.5, label='ground truth')
        plt.xlabel('Normalized Values')
        plt.ylabel('Count')
        plt.legend(loc='upper right')
        plt_name = os.path.join(model_dir, '{:05d}_histogram.png'.format(step))
        plt.savefig(plt_name, facecolor='w')
        plt.clf()
        
        # Normalized histogram
        calc = (ground_truth_video[step] - pred)/(2*(ground_truth_video[step] + pred))
        plt.hist(calc.flatten(), alpha=0.5)
        plt.xlabel('asymmetry')
        plt.ylabel('samples')
        plt.yscale('log')
        plt_name = os.path.join(model_dir, '{:05d}_asym.png'.format(step))
        plt.savefig(plt_name, facecolor='w')
        plt.clf()
        plt.close()
        
        d.append(np.sum(abs(ground_truth_video[step] - predict_video[step]), axis=None))
        st.append(np.std(predict_video[step]))
        
#     diff = np.sum(abs(ground_truth_video - predict_video), axis=(1,2,3))
    diff = np.sum(abs(ground_truth_video - predict_video), axis=(1,2))
    std = np.std(predict_video)
    ground_truth_video = np.uint8((ground_truth_video * 1.0 + 0.5) * 255)
    predict_video = np.uint8((predict_video * 1.0 + 0.5) * 255)
    render_video = np.concatenate((ground_truth_video, predict_video), axis=1)

    for step in range(predict_video.shape[0]):
        im_name = os.path.join(model_dir, '{:05d}.png'.format(step))
        gt_name = os.path.join(model_dir, '{:05d}_gt.png'.format(step))
        pred_name = os.path.join(model_dir, '{:05d}_pred.png'.format(step))

        im_render = Image.fromarray(render_video[step], 'L').convert('RGB')
        gt_im = Image.fromarray(ground_truth_video[step], 'L').convert('RGB')
        pred_im = Image.fromarray(predict_video[step], 'L').convert('RGB')
        
        gt_draw = ImageDraw.Draw(gt_im)
        pred_draw = ImageDraw.Draw(pred_im)
        draw = ImageDraw.Draw(im_render)
        draw.text((0, 0), "{:.2f}".format(diff[step]), (255,0,0))

        im_render.save(im_name)
        gt_im.save(gt_name)
        pred_im.save(pred_name)
    
    return log_pred, log_gt, d, st


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

    
    