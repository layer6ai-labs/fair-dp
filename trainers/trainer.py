import collections
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets.loaders import get_loader
from functorch import make_functional, vjp, grad
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from utils import *


class BaseTrainer:
    """Base class for various training methods"""

    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 valid_loader,
                 test_loader,
                 writer,
                 evaluator,
                 device,
                 method="regular",
                 max_epochs=100,
                 num_groups=None,
                 selected_groups=[0, 1],
                 evaluate_angles=False,
                 evaluate_hessian=False,
                 angle_comp_step=10,
                 lr=0.01,
                 seed=0,
                 num_hutchinson_estimates=100,
                 sampled_expected_loss=False
                 ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.evaluator = evaluator
        self.device = device

        self.method = method
        self.max_epochs = max_epochs
        self.num_groups = num_groups
        self.num_batch = len(self.train_loader)
        self.selected_groups = selected_groups
        self.epoch = 0
        self.num_layers = get_num_layers(self.model)

        self.evaluate_angles = evaluate_angles
        self.evaluate_hessian = evaluate_hessian
        self.angle_comp_step = angle_comp_step
        self.lr = lr
        self.seed = seed
        self.num_hutchinson_estimates = num_hutchinson_estimates
        self.sampled_expected_loss = sampled_expected_loss

    def _train_epoch(self, cosine_sim_per_epoch, expected_loss, param_for_step=None):
        # methods: regular, dpsgd, dpsgd-global, dpsgd-f, dpsgd-global-adapt
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        losses_per_group = np.zeros(self.num_groups)
        all_grad_norms = [[] for _ in range(self.num_groups)]
        group_max_grads = [0] * self.num_groups
        g_B_k_norms = [[] for _ in range(self.num_groups)]

        for _batch_idx, (data, target, group) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            losses_per_group = self.get_losses_per_group(criterion, data, target, group, losses_per_group)
            loss.backward()
            per_sample_grads = self.flatten_all_layer_params()

            # get sum of grads over groups over current batch
            if self.method == "regular":
                grad_norms, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group)
            elif self.method in ["dpsgd", "dpsgd-global", "dpsgd-global-adapt"]:
                grad_norms, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group, clipping_bound=self.optimizer.max_grad_norm)
            elif self.method == "dpsgd-f":
                C = self.compute_clipping_bound_per_sample(per_sample_grads, group)
                grad_norms, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group)
            _, group_counts_batch = split_by_group(data, target, group, self.num_groups, return_counts=1)
            g_B, g_B_k, bar_g_B, bar_g_B_k = self.mean_grads_over(group_counts_batch, sum_grad_vec_batch,
                                                                  sum_clip_grad_vec_batch)
            if (self.evaluate_angles or self.evaluate_hessian) and (
                    self.epoch * self.num_batch + _batch_idx) % self.angle_comp_step == 0:
                # compute sum of gradients over groups over entire training dataset
                if self.method == "regular":
                    sum_grad_vec_all, sum_clip_grad_vec_all, group_counts = self.get_sum_grad(
                        self.train_loader.dataset, criterion, g_B, bar_g_B, expected_loss, _batch_idx)
                elif self.method in ["dpsgd", "dpsgd-f", "dpsgd-global", "dpsgd-global-adapt"]:
                    sum_grad_vec_all, sum_clip_grad_vec_all, group_counts = self.get_sum_grad(self.train_loader.dataset,
                                                                                              criterion,
                                                                                              g_B,
                                                                                              bar_g_B,
                                                                                              expected_loss,
                                                                                              _batch_idx,
                                                                                              clipping_bound=self.optimizer.max_grad_norm)

                # average sum of gradients per group over entire training dataset
                _, g_D_k, _, _ = self.mean_grads_over(group_counts, sum_grad_vec_all, sum_clip_grad_vec_all)
                cosine_sim_per_epoch.append(self.evaluate_cosine_sim(_batch_idx, g_D_k, g_B, bar_g_B, g_B_k, bar_g_B_k))
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

            for i in range(self.num_groups):
                if len(grad_norms[i]) != 0:
                    all_grad_norms[i] = all_grad_norms[i] + grad_norms[i]
                    group_max_grads[i] = max(group_max_grads[i], max(grad_norms[i]))
                    g_B_k_norms[i].append(torch.linalg.norm(g_B_k[i]).item())

            if self.method == "dpsgd-f":
                self.optimizer.step(C)
            elif self.method == "dpsgd-global":
                self.optimizer.step(self.strict_max_grad_norm)
            elif self.method == "dpsgd-global-adapt":
                next_Z = self._update_Z(per_sample_grads, self.strict_max_grad_norm)
                self.optimizer.step(self.strict_max_grad_norm)
                self.strict_max_grad_norm = next_Z
            else:
                self.optimizer.step()
            losses.append(loss.item())
        if self.method != "regular":
            if self.method in ["dpsgd-f", "dpsgd-global-adapt"]:
                self._update_privacy_accountant()
            epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
            print(f"(ε = {epsilon:.2f}, δ = {self.delta})")
            privacy_dict = {"epsilon": epsilon, "delta": self.delta}
            self.writer.record_dict("Privacy", privacy_dict, step=0, save=True, print_results=False)
        group_ave_grad_norms = [np.mean(all_grad_norms[i]) for i in range(self.num_groups)]
        group_norm_grad_ave = [np.mean(g_B_k_norms[i]) for i in range(self.num_groups)]
        return group_ave_grad_norms, group_max_grads, group_norm_grad_ave, losses, losses_per_group / self.num_batch

    def train(self, write_checkpoint=True):
        training_time = 0
        group_loss_epochs = []
        cos_sim_per_epoch = []
        expected_loss = []
        avg_grad_norms_epochs = []
        max_grads_epochs = []
        norm_avg_grad_epochs = []
        while self.epoch < self.max_epochs:
            epoch_start_time = time.time()
            self.model.train()
            avg_grad_norms, max_grads, norm_avg_grad, losses, group_losses = self._train_epoch(cos_sim_per_epoch,
                                                                                               expected_loss)
            group_loss_epochs.append([self.epoch, np.mean(losses)] + list(group_losses))
            avg_grad_norms_epochs.append([self.epoch] + list(avg_grad_norms))
            max_grads_epochs.append([self.epoch] + list(max_grads))
            norm_avg_grad_epochs.append([self.epoch] + list(norm_avg_grad))

            epoch_training_time = time.time() - epoch_start_time
            training_time += epoch_training_time

            print(
                f"Train Epoch: {self.epoch} \t"
                f"Loss: {np.mean(losses):.6f} \t"
                f"Loss per group: {group_losses}"
            )

            self._validate()
            self.writer.write_scalar("train/" + "Loss", np.mean(losses), self.epoch)
            self.writer.write_scalars("train/AverageGrad",
                                      {'group' + str(k): v for k, v in enumerate(avg_grad_norms)},
                                      self.epoch)
            self.writer.write_scalars("train/MaxGrad",
                                      {'group' + str(k): v for k, v in enumerate(max_grads)},
                                      self.epoch)
            if write_checkpoint: self.write_checkpoint("latest")
            self.epoch += 1

            if self.epoch == self.max_epochs:
                loss_dict = dict()

                loss_dict["final_loss"] = np.mean(losses)
                loss_dict["final_loss_per_group"] = group_losses
                self.writer.record_dict("final_loss", loss_dict, 0, save=1, print_results=0)

        K = self.num_groups
        # write group_loss to csv
        columns = ["epoch", "train_loss"] + [f"train_loss_{k}" for k in range(K)]
        self.create_csv(group_loss_epochs, columns, "train_loss_per_epochs")

        # write avg_grad_norms to csv
        columns = ["epoch"] + [f"ave_grads_{k}" for k in range(K)]
        self.create_csv(avg_grad_norms_epochs, columns, "avg_grad_norms_per_epochs")

        # write max_grads_epochs to csv
        columns = ["epoch"] + [f"max_grads_{k}" for k in range(K)]
        self.create_csv(max_grads_epochs, columns, "max_grad_norms_per_epochs")

        # write norm_avg_grad to csv
        columns = ["epoch"] + [f"norm_avg_grad_{k}" for k in range(K)]
        self.create_csv(norm_avg_grad_epochs, columns, "norm_avg_grad_per_epochs")

        # write norms, angles to csv
        columns = ["epoch", "batch"] + \
                  [f"cos_g_D_{k}_g_B_{k}" for k in self.selected_groups] + \
                  [f"cos_g_D_{k}_bar_g_B_{k}" for k in self.selected_groups] + \
                  [f"cos_g_D_{k}_g_B" for k in self.selected_groups] + \
                  [f"cos_g_D_{k}_bar_g_B" for k in self.selected_groups] + \
                  ["cos_g_B_bar_g_B", "|g_B|", "|bar_g_B|"] + \
                  [f"|g_D_{k}|" for k in self.selected_groups] + \
                  [f"|g_B_{k}|" for k in self.selected_groups] + \
                  [f"|bar_g_B_{k}|" for k in self.selected_groups]
        self.create_csv(cos_sim_per_epoch, columns, "angles_per_epochs")

        # write expected loss terms to csv
        columns = ["epoch", "batch"] + \
                  [f"R_non_private_{k}" for k in self.selected_groups] + \
                  [f"R_clip_{k}" for k in self.selected_groups] + \
                  [f"R_clip_dir_inner_prod_term_{k}" for k in self.selected_groups] + \
                  [f"R_clip_dir_hess_term_{k}" for k in self.selected_groups] + \
                  [f"R_clip_dir_{k}" for k in self.selected_groups] + \
                  [f"R_clip_mag_inner_prod_term_{k}" for k in self.selected_groups] + \
                  [f"R_clip_mag_hess_term_{k}" for k in self.selected_groups] + \
                  [f"R_clip_mag_{k}" for k in self.selected_groups] + \
                  [f"R_noise_{k}" for k in self.selected_groups]
        self.create_csv(expected_loss, columns, "expected_loss_per_epochs")

        self.writer.write_scalar("train/" + "avg_train_time_over_epoch",
                                 training_time / (self.max_epochs * 60))  # in minutes
        self._test()

    def create_csv(self, data, columns, title):
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(self.writer.logdir, f"{title}.csv"), index=False)

    def flatten_all_layer_params(self):
        """
        Flatten the parameters of all layers in a model

        Args:
            model: a pytorch model

        Returns:
            a tensor of shape num_samples in a batch * num_params
        """
        per_sample_grad = None
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if per_sample_grad is None:
                    per_sample_grad = torch.flatten(p.grad_sample, 1, -1)
                else:
                    per_sample_grad = torch.cat((per_sample_grad, torch.flatten(p.grad_sample, 1, -1)), 1)
        return per_sample_grad

    def _validate(self):
        valid_results = self.evaluator.validate()
        self.writer.record_dict("Validation", valid_results, self.epoch, save=True)

    def _test(self):
        test_results = self.evaluator.test()
        self.writer.record_dict("Test", test_results, self.epoch, save=True)
        if "accuracy_per_group" in test_results.keys():
            plot_by_group(test_results["accuracy_per_group"], self.writer, data_title="final accuracy_per_group",
                          scale_to_01=1)

    def write_checkpoint(self, tag):
        checkpoint = {
            "epoch": self.epoch,

            "module_state_dict": self.model.state_dict(),
            "opt_state_dict": self.optimizer.state_dict(),
        }

        self.writer.write_checkpoint(f"{tag}", checkpoint)

    def record_expected_loss(self, R_non_private, R_clip, R_noise, R_clip_dir_inner_prod_term, R_clip_dir_hess_term,
                             R_clip_dir, R_clip_mag_inner_prod_term, R_clip_mag_hess_term, R_clip_mag, batch_idx):
        step = self.epoch * self.num_batch + batch_idx
        self.writer.write_scalars("R_non_private", {'group' + str(k): v for k, v in enumerate(R_non_private)}, step)
        self.writer.write_scalars("R_clip", {'group' + str(k): v for k, v in enumerate(R_clip)}, step)
        self.writer.write_scalars("R_noise", {'group' + str(k): v for k, v in enumerate(R_noise)}, step)
        self.writer.write_scalars("R_clip_dir_inner_prod_term",
                                  {'group' + str(k): v for k, v in enumerate(R_clip_dir_inner_prod_term)}, step)
        self.writer.write_scalars("R_clip_dir_hess_term",
                                  {'group' + str(k): v for k, v in enumerate(R_clip_dir_hess_term)}, step)
        self.writer.write_scalars("R_clip_dir", {'group' + str(k): v for k, v in enumerate(R_clip_dir)}, step)
        self.writer.write_scalars("R_clip_mag_inner_prod_term",
                                  {'group' + str(k): v for k, v in enumerate(R_clip_mag_inner_prod_term)}, step)
        self.writer.write_scalars("R_clip_mag_hess_term",
                                  {'group' + str(k): v for k, v in enumerate(R_clip_mag_hess_term)}, step)
        self.writer.write_scalars("R_clip_mag", {'group' + str(k): v for k, v in enumerate(R_clip_mag)}, step)

    def expected_loss_batch_terms(self, data, target, group, g_B, bar_g_B, C, criterion):
        def create_hvp_fn(data, target):
            func_model, params = make_functional(self.model)

            def compute_loss(params):
                preds = func_model(params, data)
                loss = criterion(preds, target)
                return loss

            _, hvp_fn = vjp(grad(compute_loss), params)
            return hvp_fn

        per_group, counts = split_by_group(data, target, group, self.num_groups, True)
        per_slct_group = [per_group[i] for i in self.selected_groups]
        slct_counts = [counts[i] for i in self.selected_groups]
        groups_len = len(self.selected_groups)
        grad_hess_grad = np.zeros(groups_len)
        clip_grad_hess_clip_grad = np.zeros(groups_len)
        R_noise = np.zeros(groups_len)
        loss = np.zeros(groups_len)
        self.model.disable_hooks()
        _, params = make_functional(self.model)
        unflattened_g_B = unflatten_grads(params, g_B)
        unflattened_bar_g_B = unflatten_grads(params, bar_g_B)
        for group_idx, (data_group, target_group) in enumerate(per_slct_group):
            with torch.no_grad():
                hvp_fn = create_hvp_fn(data_group, target_group)
                self.optimizer.zero_grad()
                preds = self.model(data_group)
                loss[group_idx] = criterion(preds, target_group) * slct_counts[group_idx]
                result = 0
                for i in range(self.num_hutchinson_estimates):
                    rand_z = tuple(rademacher(el) for el in params)
                    hess_z = hvp_fn(rand_z)[0]
                    z_hess_z = torch.sum(
                        torch.stack([torch.dot(x.flatten(), y.flatten()) for (x, y) in zip(rand_z, hess_z)]))
                    result += z_hess_z.item()
                # combine results taking into account different batch sizes
                hessian_trace = result * slct_counts[group_idx] / self.num_hutchinson_estimates
                grad_hess = hvp_fn(unflattened_g_B)[0]
                flat_grad_hess = torch.cat([torch.flatten(t) for t in grad_hess])
                grad_hess_grad[group_idx] = torch.dot(flat_grad_hess, g_B) * slct_counts[group_idx]
                clip_grad_hess = hvp_fn(unflattened_bar_g_B)[0]
                flat_clip_grad_hess = torch.cat([torch.flatten(t) for t in clip_grad_hess])
                clip_grad_hess_clip_grad[group_idx] = torch.dot(flat_clip_grad_hess, bar_g_B) * slct_counts[group_idx]
                R_noise[group_idx] = self.lr ** 2 / 2 * hessian_trace * C ** 2 * self.optimizer.noise_multiplier ** 2
        self.model.enable_hooks()
        return grad_hess_grad, clip_grad_hess_clip_grad, R_noise, loss

    def expected_loss(self, g_B, bar_g_B, sum_grad_vec, grad_hess_grad, clip_grad_hess_clip_grad,
                      R_noise, loss, group_counts, expected_loss_terms, batch_indx):
        norm_g_B = torch.linalg.norm(g_B).item()
        norm_bar_g_B = torch.linalg.norm(bar_g_B).item()
        groups_len = len(self.selected_groups)
        R_non_private = np.zeros(groups_len)
        R_clip = np.zeros(groups_len)
        new_R_clip_dir = np.zeros(groups_len)
        new_R_clip_dir_inner_prod_term = np.zeros(groups_len)
        new_R_clip_dir_hess_term = np.zeros(groups_len)
        new_R_clip_mag = np.zeros(groups_len)
        new_R_clip_mag_inner_prod_term = np.zeros(groups_len)
        new_R_clip_mag_hess_term = np.zeros(groups_len)
        for group_idx in range(groups_len):
            g_D_a = sum_grad_vec[group_idx] / group_counts[group_idx]
            group_grad_dot_grad = torch.dot(g_D_a, g_B)
            R_non_private[group_idx] = loss[group_idx] - self.lr * group_grad_dot_grad + self.lr ** 2 / 2 * \
                                       grad_hess_grad[group_idx]
            R_clip[group_idx] = self.lr * (
                    group_grad_dot_grad - torch.dot(g_D_a, bar_g_B)) \
                                + self.lr ** 2 / 2 * (clip_grad_hess_clip_grad[group_idx] - grad_hess_grad[group_idx])

            new_R_clip_dir_inner_prod_term[group_idx] = self.lr * torch.dot(g_D_a,
                                                                            norm_bar_g_B / norm_g_B * g_B - bar_g_B)
            new_R_clip_dir_hess_term[group_idx] = self.lr ** 2 / 2 * (
                    clip_grad_hess_clip_grad[group_idx] - (norm_bar_g_B / norm_g_B) ** 2 * grad_hess_grad[
                group_idx])
            new_R_clip_dir[group_idx] = new_R_clip_dir_inner_prod_term[group_idx] + new_R_clip_dir_hess_term[group_idx]
            new_R_clip_mag_inner_prod_term[group_idx] = self.lr * torch.dot(g_D_a, g_B - norm_bar_g_B / norm_g_B * g_B)
            new_R_clip_mag_hess_term[group_idx] = self.lr ** 2 / 2 * ((norm_bar_g_B / norm_g_B) ** 2 - 1) * \
                                                  grad_hess_grad[group_idx]
            new_R_clip_mag[group_idx] = new_R_clip_mag_inner_prod_term[group_idx] + new_R_clip_mag_hess_term[group_idx]

        self.record_expected_loss(R_non_private, R_clip, R_noise, new_R_clip_dir_inner_prod_term,
                                  new_R_clip_dir_hess_term,
                                  new_R_clip_dir, new_R_clip_mag_inner_prod_term, new_R_clip_mag_hess_term,
                                  new_R_clip_mag, batch_indx)
        row = [self.epoch,
               batch_indx] + R_non_private.tolist() + R_clip.tolist() + new_R_clip_dir_inner_prod_term.tolist() + \
              new_R_clip_dir_hess_term.tolist() + new_R_clip_dir.tolist() + new_R_clip_mag_inner_prod_term.tolist() + \
              new_R_clip_mag_hess_term.tolist() + new_R_clip_mag.tolist() + R_noise.tolist()
        expected_loss_terms.append(row)

    def get_losses_per_group(self, criterion, data, target, group, group_losses):
        '''
        Given subset of GroupLabelDataset (data, target, group), computes
        loss of model on each subset (data, target, group=k) and returns
        np array of length num_groups = group_losses + group losses over given data
        '''
        per_group = split_by_group(data, target, group, self.num_groups)
        group_loss_batch = np.zeros(self.num_groups)
        for group_idx, (data_group, target_group) in enumerate(per_group):
            with torch.no_grad():
                if data_group.shape[0] == 0:  # if batch does not contain samples of group i
                    group_loss_batch[group_idx] = 0
                else:
                    group_output = self.model(data_group)
                    group_loss_batch[group_idx] = criterion(group_output, target_group).item()
        group_losses = group_loss_batch + group_losses
        return group_losses

    def get_sum_grad_batch(self, data, targets, groups, criterion, **kwargs):
        data = data.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        per_sample_grads = self.flatten_all_layer_params()

        return self.get_sum_grad_batch_from_vec(per_sample_grads, groups, **kwargs)

    def get_sum_grad_batch_from_vec(self, per_sample_grads, groups, **kwargs):
        if self.method == "dpsgd-f":
            clipping_bounds = self.compute_clipping_bound_per_sample(per_sample_grads, groups)
            grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec = get_grad_norms_clip(per_sample_grads, groups,
                                                                                               self.num_groups,
                                                                                               self.clipping_scale_fn,
                                                                                               clipping_bounds=clipping_bounds)
        else:
            grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec = get_grad_norms_clip(per_sample_grads, groups,
                                                                                               self.num_groups,
                                                                                               self.clipping_scale_fn,
                                                                                               **kwargs)
        return grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec

    def get_sum_grad(self, dataset, criterion, g_B, bar_g_B, expected_loss_terms, batch_idx, **kwargs):
        loader = get_loader(self.train_loader.dataset, self.device, 1000, drop_last=False)
        groups_len = len(self.selected_groups)
        running_sum_grad_vec = None
        running_sum_clip_grad_vec = None
        sum_grad_hess_grad = np.zeros(groups_len)
        sum_clip_grad_hess_clip_grad = np.zeros(groups_len)
        sum_R_noise = np.zeros(groups_len)
        sum_loss = np.zeros(groups_len)
        # First argument is a dummy
        _, group_counts = split_by_group(dataset.y, dataset.y, dataset.z, self.num_groups, return_counts=True)
        for data, target, group in loader:
            if self.method == "dpsgd-f":
                _, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch(
                    data, target, group, criterion, **kwargs)
            else:
                _, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch(
                    data, target, group, criterion, **kwargs)
            if running_sum_grad_vec is None:
                running_sum_grad_vec = sum_grad_vec_batch
            else:
                running_sum_grad_vec = [a + b for a, b in zip(running_sum_grad_vec, sum_grad_vec_batch)]
            if running_sum_clip_grad_vec is None:
                running_sum_clip_grad_vec = sum_clip_grad_vec_batch
            else:
                running_sum_clip_grad_vec = [a + b for a, b in zip(running_sum_clip_grad_vec, sum_clip_grad_vec_batch)]
            if self.evaluate_hessian and self.method != "regular":
                clipping_bound = kwargs['clipping_bound']
                grad_hess_grad, clip_grad_hess_clip_grad, R_noise, loss = self.expected_loss_batch_terms(
                    data, target, group, g_B, bar_g_B, clipping_bound, criterion)
                sum_grad_hess_grad += grad_hess_grad
                sum_clip_grad_hess_clip_grad += clip_grad_hess_clip_grad
                sum_R_noise += R_noise
                sum_loss += loss
            if self.sampled_expected_loss:
                _, group_counts = split_by_group(data, target, group, self.num_groups, return_counts=True)
                break


        if self.evaluate_hessian:
            final_sum_grad_vec_batch = [running_sum_grad_vec[i] for i in self.selected_groups]
            group_counts_vec = np.array([group_counts[i] for i in self.selected_groups])
            final_grad_hess_grad = sum_grad_hess_grad / group_counts_vec
            final_clip_grad_hess_clip_grad = sum_clip_grad_hess_clip_grad / group_counts_vec
            final_R_noise = sum_R_noise / group_counts_vec
            final_loss = sum_loss / group_counts_vec
            self.expected_loss(g_B, bar_g_B, final_sum_grad_vec_batch, final_grad_hess_grad,
                               final_clip_grad_hess_clip_grad, final_R_noise, final_loss,
                               group_counts_vec, expected_loss_terms, batch_idx)
        return running_sum_grad_vec, running_sum_clip_grad_vec, group_counts

    def mean_grads_over(self, group_counts, sum_grad_vec, clip_sum_grad_vec):
        g_D = torch.stack(sum_grad_vec, dim=0).sum(dim=0) / sum(group_counts)
        g_D_k = [sum_grad_vec[i] / group_counts[i] for i in range(self.num_groups)]

        bar_g_D = torch.stack(clip_sum_grad_vec, dim=0).sum(dim=0) / sum(group_counts)
        bar_g_D_k = [clip_sum_grad_vec[i] / group_counts[i] for i in range(self.num_groups)]
        return g_D, g_D_k, bar_g_D, bar_g_D_k

    def evaluate_cosine_sim(self, batch_idx, g_D_k, g_B, bar_g_B, g_B_k, bar_g_B_k):
        cos_g_D_k_g_B_k = []
        cos_g_D_k_bar_g_B_k = []
        cos_g_D_k_g_B = []
        cos_g_D_k_bar_g_B = []
        norm_g_D_k = []
        norm_g_B_k = []
        norm_bar_g_B_k = []

        cos_g_B_bar_g_B = cosine_similarity(g_B, bar_g_B, dim=0).item()
        norm_g_B = torch.linalg.norm(g_B).item()
        norm_bar_g_B = torch.linalg.norm(bar_g_B).item()

        for k in self.selected_groups:
            cos_g_D_k_g_B_k.append(cosine_similarity(g_D_k[k], g_B_k[k], dim=0).item())
            cos_g_D_k_bar_g_B_k.append(cosine_similarity(g_D_k[k], bar_g_B_k[k], dim=0).item())
            cos_g_D_k_g_B.append(cosine_similarity(g_D_k[k], g_B, dim=0).item())
            cos_g_D_k_bar_g_B.append(cosine_similarity(g_D_k[k], bar_g_B, dim=0).item())

            norm_g_D_k.append(torch.linalg.norm(g_D_k[k]).item())
            norm_g_B_k.append(torch.linalg.norm(g_B_k[k]).item())
            norm_bar_g_B_k.append(torch.linalg.norm(bar_g_B_k[k]).item())

        row = [self.epoch, batch_idx] + cos_g_D_k_g_B_k + cos_g_D_k_bar_g_B_k + cos_g_D_k_g_B + cos_g_D_k_bar_g_B + [
            cos_g_B_bar_g_B, norm_g_B, norm_bar_g_B] + norm_g_D_k + norm_g_B_k + norm_bar_g_B_k
        return row


class RegularTrainer(BaseTrainer):
    """Class for non-private training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx):
        return 1


class DpsgdTrainer(BaseTrainer):
    """Class for DPSGD training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        return min(1, clipping_bound / grad_norm)

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta


class DpsgdFTrainer(BaseTrainer):
    """Class for DPSGD-F training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, **kwargs):
        clipping_bounds = kwargs["clipping_bounds"]
        return min((clipping_bounds[idx] / grad_norm).item(), 1)

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            base_max_grad_norm=1,  # C0
            counts_noise_multiplier=10,  # noise multiplier applied on mk and ok
            **kwargs
    ):
        """
        Initialization function. Initialize parent class while adding new parameter clipping_bound and noise_scale.

        Args:
            model: model from privacy_engine.make_private()
            optimizer: a DPSGDF_Optimizer
            privacy_engine: DPSGDF_Engine
            train_loader: train_loader from privacy_engine.make_private()
            valid_loader: normal pytorch data loader for validation set
            test_loader: normal pytorch data loader for test set
            writer: writer to tensorboard
            evaluator: evaluate for model performance
            device: device to train the model
            delta: definition in privacy budget
            clipping_bound: C0 in the original paper, defines the threshold of gradients
            counts_noise_multiplier: sigma1 in the original paper, defines noise added to the number of samples with gradient bigger than clipping_bound C0
        """
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta
        # new parameters for DPSGDF
        self.base_max_grad_norm = base_max_grad_norm  # C0
        self.counts_noise_multiplier = counts_noise_multiplier  # noise scale applied on mk and ok
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def compute_clipping_bound_per_sample(self, per_sample_grads, group):
        """compute clipping bound for each sample """
        # calculate mk, ok
        mk = collections.defaultdict(int)
        ok = collections.defaultdict(int)
        # get the l2 norm of gradients of all parameters for each sample, in shape of (batch_size, )
        l2_norm_grad_per_sample = torch.norm(per_sample_grads, p=2, dim=1)  # batch_size

        assert len(group) == len(l2_norm_grad_per_sample)

        for i in range(len(group)):  # looping over batch
            group_idx = group[i].item()
            if l2_norm_grad_per_sample[i].item() > self.base_max_grad_norm:
                mk[group_idx] += 1
            else:
                ok[group_idx] += 1

        # add noise scale to mk and ok
        m2k = {}
        o2k = {}
        m = 0

        # note that some group idx might have 0 sample counts in the batch and we are still adding noise to it
        for group_idx in range(self.num_groups):
            m2k[group_idx] = mk[group_idx] + torch.normal(0, self.counts_noise_multiplier, (1,)).item()
            m2k[group_idx] = max(int(m2k[group_idx]), 0)
            o2k[group_idx] = ok[group_idx] + torch.normal(0, self.counts_noise_multiplier, (1,)).item()
            o2k[group_idx] = max(int(o2k[group_idx]), 0)
            m += m2k[group_idx]

        # Account for privacy cost of privately estimating group sizes
        # using the built in sampled-gaussian-mechanism accountant.
        # L2 sensitivity of per-group counts vector is always 1,
        # so std use in torch.normal is the same as noise_multiplier in accountant.
        # Accounting is done lazily, see _update_privacy_accountant method.
        self.privacy_step_history.append([self.counts_noise_multiplier, self.sample_rate])
        array = []
        bk = {}
        Ck = {}
        for group_idx in range(self.num_groups):
            bk[group_idx] = m2k[group_idx] + o2k[group_idx]
            # added
            if bk[group_idx] == 0:
                array.append(1)  # when bk = 0, m2k = 0, we have 0/0 = 1
            else:
                array.append(m2k[group_idx] * 1.0 / bk[group_idx])

        for group_idx in range(self.num_groups):
            Ck[group_idx] = self.base_max_grad_norm * (1 + array[group_idx] / (np.mean(array) + 1e-8))

        per_sample_clipping_bound = []
        for i in range(len(group)):  # looping over batch
            group_idx = group[i].item()
            per_sample_clipping_bound.append(Ck[group_idx])

        return torch.Tensor(per_sample_clipping_bound).to(device=self.device)


class DpsgdGlobalTrainer(DpsgdTrainer):

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        if grad_norm > self.strict_max_grad_norm:
            return 0
        else:
            return clipping_bound / self.strict_max_grad_norm

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            strict_max_grad_norm=100,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=delta,
            **kwargs
        )
        self.strict_max_grad_norm = strict_max_grad_norm


class DpsgdGlobalAdaptiveTrainer(BaseTrainer):

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        if grad_norm > self.strict_max_grad_norm:
            return min(1, clipping_bound / grad_norm)
        else:
            return clipping_bound / self.strict_max_grad_norm

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=1e-5,
            strict_max_grad_norm=100,
            bits_noise_multiplier=10,
            lr_Z=0.01,
            threshold=1.0,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )
        self.privacy_engine = privacy_engine
        self.delta = delta
        self.strict_max_grad_norm = strict_max_grad_norm  # Z
        self.bits_noise_multiplier = bits_noise_multiplier
        self.lr_Z = lr_Z
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []
        self.threshold = threshold

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def _update_Z(self, per_sample_grads, Z):
        # get the l2 norm of gradients of all parameters for each sample, in shape of (batch_size, )
        l2_norm_grad_per_sample = torch.norm(per_sample_grads, p=2, dim=1)
        batch_size = len(l2_norm_grad_per_sample)

        dt = 0  # sample count in a batch exceeding Z * threshold
        for i in range(batch_size):  # looping over batch
            if l2_norm_grad_per_sample[i].item() > self.threshold * Z:
                dt += 1

        dt = dt * 1.0 / batch_size  # percentage of samples in a batch that's bigger than the threshold * Z
        noisy_dt = dt + torch.normal(0, self.bits_noise_multiplier, (1,)).item() * 1.0 / batch_size

        factor = math.exp(- self.lr_Z + noisy_dt)

        next_Z = Z * factor

        self.privacy_step_history.append([self.bits_noise_multiplier, self.sample_rate])
        return next_Z
