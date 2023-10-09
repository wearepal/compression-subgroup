import os
import torch


# Defined classes
class LossComputer:
    
    def __init__(self, dataset, criterion, gamma=0.1, adj=None):

        self.criterion = criterion
        self.gamma = gamma

        self.n_classes = dataset.n_classes

        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().cuda()
        self.group_frac = self.group_counts / self.group_counts.sum()
        self.group_str = dataset.group_str
        
        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        # Quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda() / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def reset_stats(self):
        
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    def loss(self, yhat, y, group_idx=None):
        
        # Compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)

        # Update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # Compute overall loss
        actual_loss = per_sample_losses.mean()
        weights = None

        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
        return actual_loss

    def compute_group_avg(self, losses, group_idx):
        
        # Compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss

        # counts
        self.processed_data_counts += group_count
        self.update_data_counts += group_count
        self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        
        model_norm_sq = 0.0
        for param in model.parameters():
            model_norm_sq += torch.norm(param)**2
        
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args['weight_decay'] / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, path, model):

        # Get the model size in MB
        temp_path = os.path.join(path, 'temp.p')
        torch.save(model.state_dict(), temp_path)
        size_all_mb = os.path.getsize(temp_path) / 1e6
        os.remove(temp_path)
        
        with open(os.path.join(path, 'results.txt'), 'w') as f:
            
            print(f'\nAverage incurred loss: {self.avg_per_sample_loss.item():.3f}', file=f)
            print(f'Average sample loss: {self.avg_actual_loss.item():.3f}\n', file=f)
            
            print(f'Average acc: {self.avg_acc.item():.3f}', file=f)
            print(f'Worst acc: {min(self.avg_group_acc):.3f}', file=f)
            print(f'Model size: {size_all_mb:.2f}\n', file=f)
            
            for group_idx in range(self.n_groups):
                
                print(
                    f'{self.group_str(group_idx)} '
                    f'[n = {int(self.processed_data_counts[group_idx])}]: '
                    f'loss = {self.avg_group_loss[group_idx]:.3f} '
                    f'exp loss = {self.exp_avg_loss[group_idx]:.3f} '
                    f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f} '
                    f'adv prob = {self.adv_probs[group_idx]:3f} '
                    f'acc = {self.avg_group_acc[group_idx]:.3f}\n'
                , file=f)
                
        with open(os.path.join(path, 'results.txt'), 'r') as f:
            print(f.read())
