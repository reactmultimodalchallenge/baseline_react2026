import math
import torch
import time
from einops import rearrange
from tqdm import tqdm
from utils.meters import AverageMeter
from utils.loss import DistributionLoss, AllThreMseLoss, MidLoss
# PCC; PearsonCC; AllMseLoss; ThreMseLoss; S_MSE
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_eeg_metrics, compute_metrics


class Trainer(object):
    def __init__(self, model, loss_name='MSE', cal_logdets=True, no_inverse=False,
                 neighbor_pattern='nearest', num_frames=50, loss_mid=False, num_preds=10, batch_size=16,
                 train_eeg_head_only=False, eeg_loss_weight=0.25, eval_eeg=False,
                 metric_threads=1, skip_facial_metrics=False, save_results=True):

        self.model = model
        if loss_name == 'Distribution':
            self.criterion = DistributionLoss()
        else:
            print('Use Neighbor Pattern: ' + neighbor_pattern)
            if neighbor_pattern == 'nearest':
                self.threshold = None
                self.criterion = AllThreMseLoss()
                self.mid_criterion = MidLoss()
            elif neighbor_pattern == 'all':
                self.threshold = None
                self.criterion = AllThreMseLoss(cal_type='min')
                self.mid_criterion = MidLoss(loss_type='L2')

        self.cal_logdets = cal_logdets
        self.no_inverse = no_inverse
        self.neighbor_pattern = neighbor_pattern
        self.num_frames = num_frames
        self.loss_mid = loss_mid
        self.num_preds = num_preds
        self.train_eeg_head_only = train_eeg_head_only
        self.eeg_loss_weight = eeg_loss_weight
        self.eval_eeg = eval_eeg
        self.metric_threads = metric_threads
        self.skip_facial_metrics = skip_facial_metrics
        self.save_results = save_results
        self.data_parser = None
        if not train_eeg_head_only and not skip_facial_metrics:
            self.data_parser = Processor(
                ckpt_dir="../pretrained_models/post_processor",
                num_preds=num_preds,
                cfg_dir="../",
            )
        self.bsz = max(int(batch_size), 1)

    @staticmethod
    def masked_mse(prediction, target, mask):
        mask = mask.to(dtype=prediction.dtype)
        loss = ((prediction - target) ** 2) * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    def _segment_torch2d(self, tensor):
        length, dim = tensor.size()
        pad = (-length) % self.num_frames
        if pad:
            tensor = torch.cat((tensor, tensor.new_zeros(pad, dim)), dim=0)
        return tensor.view(-1, self.num_frames, dim)

    def _parse_train_speaker_emotion(self, speaker_param_inputs, indices):
        speaker_emotions = [
            speaker_param_input[index:index + self.num_frames]
            for speaker_param_input, index in zip(speaker_param_inputs, indices)
        ]
        return torch.stack(speaker_emotions, dim=0)

    def _parse_test_speaker_emotion(self, speaker_param_inputs):
        speaker_emotions = [self._segment_torch2d(speaker_param_input) for speaker_param_input in speaker_param_inputs]
        return torch.cat(speaker_emotions, dim=0)

    def _parse_data(self,
                    speaker_video_inputs,
                    speaker_audio_inputs,
                    listener_param_gts,
                    indices,
                    seq_lens,
                    test=False):
        if self.data_parser is None:
            raise RuntimeError("Data parser is disabled for this trainer mode.")
        inputs = [torch.zeros(size=(seq_len, 25)) for seq_len in seq_lens]
        listener_param_gts = self.data_parser.forward(inputs, listener_param_gts)

        if test:
            # speaker_video_inputs -> List: [Tensor([N, n_frames, d_v]), Tensor([N', n_frames, d_v]), ...]'])]
            # listener_param_gts -> List: [Tensor([num_preds, l, d_e]), Tensor([num_preds, l', d_e]), ...]
            num_clip_list = [s.shape[0] for s in speaker_video_inputs]
            speaker_video_inputs = torch.cat(speaker_video_inputs, dim=0)  # [B, n_frames, d_v]
            speaker_audio_inputs = torch.cat(speaker_audio_inputs, dim=0)  # [B, n_frames, d_a]
            lengths = (seq_lens, num_clip_list)
        else:
            # speaker_video_inputs, speaker_audio_inputs -> List: [Tensor([n_frames, d]), ...]
            speaker_audio_inputs = torch.stack(speaker_audio_inputs, dim=0)  # [bsz, n_frames, d_v]
            speaker_video_inputs = torch.stack(speaker_video_inputs, dim=0)  # [bsz, n_frames, d_a]

            lengths = []
            listener_param_gts_list = []
            for idx, l_param_gt in zip(indices, listener_param_gts):
                listener_param_gts_list.append(l_param_gt[:, idx:idx + self.num_frames])
                lengths.append(l_param_gt.shape[0])
            listener_param_gts = torch.cat(listener_param_gts_list, dim=0).transpose(2, 1)
            lengths = torch.tensor(lengths)
            # listener_param_gts = [l_param_gt[:, idx:idx + self.num_frames] \
            #                       for idx, l_param_gt in zip(indices, listener_param_gts)]
            # lengths = torch.tensor([self.num_preds] * len(speaker_video_inputs))
            # [bsz * num_preds, n_frames, d_e]

        return speaker_video_inputs, speaker_audio_inputs, listener_param_gts, lengths

    def _parse_test_inputs(self, speaker_video_inputs, speaker_audio_inputs, seq_lens):
        num_clip_list = [speaker_video_input.shape[0] for speaker_video_input in speaker_video_inputs]
        speaker_video_inputs = torch.cat(speaker_video_inputs, dim=0)
        speaker_audio_inputs = torch.cat(speaker_audio_inputs, dim=0)
        return speaker_video_inputs, speaker_audio_inputs, (seq_lens, num_clip_list)

    def train(self, epoch, dataloader, optimizer, print_freq=1, train_iters=100):
        if self.train_eeg_head_only:
            self.model.set_eeg_head_train_mode()
        else:
            self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        process_time = AverageMeter()
        losses_dtw = AverageMeter()
        losses_mid = AverageMeter()
        losses_det = AverageMeter()
        losses_eeg = AverageMeter()
        eeg_valid_ratios = AverageMeter()
        since = time.time()

        for i, batch in tqdm(enumerate(dataloader)):
            e_inputs, v_inputs, a_inputs, targets, indices, num_frames = batch[:6]
            listener_eeg_targets = listener_eeg_masks = None
            if len(batch) > 6:
                listener_eeg_targets, listener_eeg_masks = batch[6:8]
            data_time.update(time.time() - since)

            if self.train_eeg_head_only:
                if listener_eeg_targets is None or listener_eeg_masks is None:
                    raise RuntimeError("EEG head-only training requires EEG labels from the dataloader.")

                v_inputs = torch.stack(v_inputs, dim=0).cuda()
                a_inputs = torch.stack(a_inputs, dim=0).cuda()
                speaker_emotions = self._parse_train_speaker_emotion(e_inputs, indices).cuda()
                listener_eeg_targets = torch.stack(listener_eeg_targets, dim=0).cuda().float()
                listener_eeg_masks = torch.stack(listener_eeg_masks, dim=0).cuda().float()
                process_time.update(time.time() - since)

                with torch.no_grad():
                    predicted_emotions = self.model.inverse(
                        v_inputs,
                        a_inputs,
                        cal_norm=True,
                        threshold=self.threshold,
                    ).detach()
                prediction_eeg = self.model.predict_eeg(
                    v_inputs,
                    a_inputs,
                    speaker_emotions,
                    predicted_emotions,
                )
                loss_eeg = self.masked_mse(prediction_eeg, listener_eeg_targets, listener_eeg_masks)
                eeg_valid_ratio = listener_eeg_masks.float().mean()

                optimizer.zero_grad()
                loss_eeg.backward()
                optimizer.step()

                losses_eeg.update(loss_eeg.item())
                eeg_valid_ratios.update(eeg_valid_ratio.item())
                batch_time.update(time.time() - since)
                since = time.time()

                if ((i + 1) % print_freq == 0):
                    print('Epoch: [{}][{}/{}]\t'
                          'Whole Time {:.3f} ({:.3f})\t'
                          'Process Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          'Loss_EEG {:.6f} ({:.6f})\t'
                          'EEG_valid_ratio {:.4f} ({:.4f})\t'
                          .format(epoch, i + 1, train_iters,
                                  batch_time.val, batch_time.avg,
                                  process_time.val, process_time.avg,
                                  data_time.val, data_time.avg,
                                  losses_eeg.val, losses_eeg.avg,
                                  eeg_valid_ratios.val, eeg_valid_ratios.avg,
                                  ))
                    print('-' * 160)

                if i == train_iters:
                    break
                continue

            v_inputs, a_inputs, targets, lengths = self._parse_data(
                v_inputs, a_inputs, targets, indices, num_frames, test=False,
            )
            # v_inputs: [bsz, n_frames, d_v]; a_inputs: [bsz, n_frames, d_a]; targets: [bsz * num_preds, n_frames, d_e]
            v_inputs, a_inputs, targets, lengths = v_inputs.cuda(), a_inputs.cuda(), targets.cuda(), lengths.cuda()
            process_time.update(time.time() - since)

            torch.autograd.set_detect_anomaly(True)
            if not self.no_inverse:
                speaker_features, listener_features, params, edge, nearest_targets, loss_det = \
                    self.model(v_inputs, a_inputs, targets, lengths)
            else:
                # Perform alignment after the inversion of samples
                speaker_features, listener_features, loss_det = self.model(v_inputs, a_inputs, targets, lengths)

            if self.neighbor_pattern == 'all':
                loss_dtw = self.criterion(speaker_features, listener_features, lengths, threshold=self.threshold)
                if self.loss_mid:
                    loss_mid = self.mid_criterion(listener_features, lengths)
                else:
                    loss_mid = 0.0
            else:
                loss_dtw = self.criterion(speaker_features, listener_features, lengths, threshold=self.threshold)

            loss = loss_dtw + (loss_det if self.cal_logdets else 0.) + loss_mid

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_dtw.update(loss_dtw.item())
            losses_mid.update(loss_mid.item() if not type(loss_mid) == float else 0.)
            losses_det.update(loss_det.item() if not type(loss_det) == float else 0.)
            batch_time.update(time.time() - since)
            since = time.time()

            torch.set_printoptions(precision=4, sci_mode=False)
            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Whole Time {:.3f} ({:.3f})\t'
                      'Process Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_DTW {:.3f} ({:.3f})\t'
                      'Loss_MID {:.3f} ({:.3f})\t'
                      'Loss_DET {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              process_time.val, process_time.avg,
                              data_time.val, data_time.avg,
                              losses_dtw.val, losses_dtw.avg,
                              losses_mid.val, losses_mid.avg,
                              losses_det.val, losses_det.avg,
                              ))
                print('-' * 160)

            if i == train_iters:
                self.threshold = torch.tensor([(losses_dtw.avg / 3)]).cuda()
                print('='*10 + ' threshold ' + '='*10)
                print(self.threshold)
                break


    def test(self, testloader, modify=False):
        self.model.eval()
        test_iters = len(testloader)
        print(f"Length of testloader: {test_iters}")
        self.modify = modify

        keep_facial_outputs = not self.skip_facial_metrics
        speaker_params_all = []
        predictions_all = []
        targets_all = []
        listener_eeg_preds_all = []
        listener_eeg_targets_all = []
        listener_eeg_masks_all = []
        for bsz_idx, batch in tqdm(
                enumerate(testloader),
                total=test_iters,
                desc="Evaluating RegNN",
                unit="sample",
                dynamic_ncols=True):
            e_inputs, v_inputs, a_inputs, targets, indices, num_frames = batch[:6]
            listener_eeg_targets = listener_eeg_masks = None
            if len(batch) > 6:
                listener_eeg_targets, listener_eeg_masks = batch[6:8]
            # TODO debug: ==========
            # if bsz_idx > 5:
            #     break
            # TODO debug: ==========

            if keep_facial_outputs:
                speaker_params_all.extend(e_inputs)
                v_inputs, a_inputs, targets, (seq_lens, num_clip_list) = self._parse_data(
                    v_inputs, a_inputs, targets, indices, num_frames, test=True,
                )
            else:
                v_inputs, a_inputs, (seq_lens, num_clip_list) = self._parse_test_inputs(
                    v_inputs, a_inputs, num_frames,
                )
            speaker_emotions = self._parse_test_speaker_emotion(e_inputs)
            v_inputs, a_inputs = v_inputs.cuda(), a_inputs.cuda()
            speaker_emotions = speaker_emotions.cuda()

            SAMPLE_NUMS = self.num_preds
            pred_list = []
            eeg_pred_list = []
            num_micro_batches = math.ceil(v_inputs.shape[0] / self.bsz)
            sample_progress = tqdm(
                total=SAMPLE_NUMS * num_micro_batches,
                desc=f"Sample {bsz_idx + 1}/{test_iters}",
                unit="clip",
                leave=False,
                dynamic_ncols=True,
            )
            with torch.inference_mode():
                try:
                    for k in range(SAMPLE_NUMS):
                        prediction_chunks = []
                        eeg_prediction_chunks = []
                        for i in range(num_micro_batches):
                            v_input_data = v_inputs[i*self.bsz:(i+1)*self.bsz]  # [bsz, n_frames, d_v]
                            a_input_data = a_inputs[i*self.bsz:(i+1)*self.bsz]  # [bsz, n_frames, d_a]
                            speaker_emotion_data = speaker_emotions[i*self.bsz:(i+1)*self.bsz]

                            pred_cuda = self.model.inverse(v_input_data, a_input_data,
                                                           cal_norm=True, threshold=self.threshold)
                            if self.modify:
                                pred_cuda[:, :, :15] = torch.where(
                                    pred_cuda[:, :, :15] >= 1.0,
                                    torch.tensor(1.0, device=pred_cuda.device),
                                    torch.tensor(0.0, device=pred_cuda.device),
                                )
                            if self.eval_eeg:
                                eeg_prediction_chunks.append(
                                    self.model.predict_eeg(
                                        v_input_data,
                                        a_input_data,
                                        speaker_emotion_data,
                                        pred_cuda,
                                    ).detach().cpu()
                                )
                            if keep_facial_outputs:
                                prediction_chunks.append(pred_cuda.detach().cpu())
                            del pred_cuda
                            sample_progress.update(1)

                        indices = [0] + list(np.cumsum(num_clip_list))
                        if keep_facial_outputs:
                            predictions = torch.cat(prediction_chunks, dim=0)
                            for j, (start_idx, end_idx) in enumerate(zip(indices[:-1], indices[1:])):
                                pred = predictions[start_idx:end_idx]
                                pred = rearrange(pred, 'b l d -> (b l) d')[:seq_lens[j]].unsqueeze(0)  # [1, L, d_e]

                                if k == 0:
                                    pred_list.append(pred)
                                else:
                                    pred_list[j] = torch.cat((pred_list[j], pred), dim=0)  # [num_preds, L, d_e]

                        if self.eval_eeg:
                            eeg_predictions = torch.cat(eeg_prediction_chunks, dim=0)
                            for j, (start_idx, end_idx) in enumerate(zip(indices[:-1], indices[1:])):
                                eeg_pred = eeg_predictions[start_idx:end_idx].unsqueeze(0)
                                if k == 0:
                                    eeg_pred_list.append(eeg_pred)
                                else:
                                    eeg_pred_list[j] = torch.cat((eeg_pred_list[j], eeg_pred), dim=0)
                finally:
                    sample_progress.close()

            if keep_facial_outputs:
                predictions_all.extend(pred_list)
                targets_all.extend(targets)
            if self.eval_eeg:
                if listener_eeg_targets is None or listener_eeg_masks is None:
                    raise RuntimeError("EEG evaluation requires EEG labels from the dataloader.")
                listener_eeg_preds_all.extend(eeg_pred_list)
                listener_eeg_targets_all.extend(target.unsqueeze(0) for target in listener_eeg_targets)
                listener_eeg_masks_all.extend(mask.unsqueeze(0) for mask in listener_eeg_masks)
            # List: [Tensor([num_preds, L, d_e]), Tensor([num_preds, L', d_e]), ...]

        # TODO saving Tensor List
        if self.save_results:
            try:
                result_dict = {}
                if keep_facial_outputs:
                    result_dict.update({'GT': targets_all, 'PRED': predictions_all})
                if self.eval_eeg:
                    result_dict.update({
                        'GT_EEG': listener_eeg_targets_all,
                        'PRED_EEG': listener_eeg_preds_all,
                        'EEG_MASK': listener_eeg_masks_all,
                    })
                torch.save(result_dict, 'results.pt')
                print("Successfully saved Tensor List")
            except Exception:
                print("Failed to save Tensor List")

        results = {}
        if not self.skip_facial_metrics:
            results.update(compute_metrics(
                speaker_params_all,
                predictions_all,
                targets_all,
                threads=self.metric_threads,
            ))
        if self.eval_eeg:
            results.update(compute_eeg_metrics(
                listener_eeg_preds_all,
                listener_eeg_targets_all,
                listener_eeg_masks_all,
            ))

        print(results)
