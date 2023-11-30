import numpy as np
from typing import Dict, List, Optional

import torch
import pytorch_lightning as pl

from src import utils
from src.dynamics_gvp import DynamicsWithPockets
from src.edm import AREDM
from src.datasets import HierCrossDockDataset, get_dataloader, collate_pocket

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception('activation fn not found. add it here')

class AR_DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}
    FRAMES = 50
    def __init__(
            self,
            lig_nf, # total number of features for ligand (atom types (10) 
            pocket_nf, # number of node features: (atom types (4) + AA type (20) + BB (1)) 
            n_dims,
            context_node_nf, # number of context features (2 or 1)
            hidden_nf,
            joint_nf,
            activation,
            tanh,
            n_layers,
            attention,
            normalization_factor, # 100
            diffusion_steps, # 500
            diffusion_noise_schedule, # learned
            diffusion_noise_precision, # 1e-5
            diffusion_loss_type, # l2
            normalize_factors, # [1, 4, 10]
            include_charges, # False
            data_path,
            train_data_prefix,
            val_data_prefix,
            batch_size,
            lr, 
            test_epochs,
            n_stability_samples,
            dataset_type,
            use_context=True,
            clip_grad=False,
            log_iterations=50,
            samples_dir=None,
            data_augmentation=False,
            center_of_mass='anchors',
            inpainting=False,
            anchors_context=True,
            n_hierarchical_steps=10,
            train_dataframe_path='paths_train.csv', # path to dataframe 
            val_dataframe_path = 'paths_val.csv',
            use_anchors_generation=True, # whether to use anchors during generation
            num_workers=0,
            edge_cutoff_ligand=None,
            edge_cutoff_pocket=4.5,
            edge_cutoff_interaction=4.5
    ):
        super(AR_DDPM, self).__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting
        self.loss_type = diffusion_loss_type
        self.use_context = use_context

        self.n_dims = n_dims
        self.include_charges = include_charges
        self.anchors_context = anchors_context
        self.n_hierarchical_steps = n_hierarchical_steps
        self.train_dataframe_path = train_dataframe_path
        self.val_dataframe_path = val_dataframe_path
        self.use_anchors_generation = use_anchors_generation
        self.num_workers = num_workers
        self.dataset_type = dataset_type
        self.clip_grad = clip_grad

        self.joint_nf = joint_nf

        self.edge_cutoff_ligand = edge_cutoff_ligand
        self.edge_cutoff_pocket = edge_cutoff_pocket
        self.edge_cutoff_interaction = edge_cutoff_interaction

        if type(activation) is str:
            activation = get_activation(activation)
        
        if self.dataset_type == 'CrossDock':
            dynamics_class = DynamicsWithPockets
        else:
            raise ValueError
    
        dynamics = dynamics_class(
            n_dims=n_dims,
            lig_nf=lig_nf,
            pocket_nf=pocket_nf,
            joint_nf=joint_nf,
            context_node_nf=context_node_nf,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            normalization_factor=normalization_factor,
            edge_cutoff_ligand=edge_cutoff_ligand,
            edge_cutoff_pocket=edge_cutoff_pocket,
            edge_cutoff_interaction=edge_cutoff_interaction
        )

        self.edm = AREDM(
                dynamics=dynamics,
                lig_nf=lig_nf,
                pocket_nf=pocket_nf,
                n_dims=n_dims,
                timesteps=diffusion_steps,
                noise_schedule=diffusion_noise_schedule,
                noise_precision=diffusion_noise_precision,
                loss_type=diffusion_loss_type,
                norm_values=normalize_factors,
                n_hier_steps=n_hierarchical_steps,
            ) 
    
    def setup(self, stage: Optional[str]=None):
        if stage == 'fit':
            self.train_dataset = HierCrossDockDataset(
                data_path=self.data_path,
                prefix=self.train_data_prefix,
                device=self.device,
                dataframe_path=self.train_dataframe_path
            )
            print('loaded train data')
            self.val_dataset = HierCrossDockDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.device,
                dataframe_path=self.val_dataframe_path
            )
            print('loaded validation data')
        elif stage == 'val':
            self.val_dataset = HierCrossDockDataset(
                data_path=self.data_path,
                prefix=self.val_data_prefix,
                device=self.device,
                dataframe_path=self.val_dataframe_path
            )
        else:
            raise NotImplementedError
    
    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_size, num_workers=self.num_workers, collate_fn=collate_pocket, shuffle=True)
        
    def val_dataloader(self):
        return get_dataloader(self.val_dataset, self.batch_size, num_workers=self.num_workers, collate_fn=collate_pocket)

    def test_dataloader(self):
        return get_dataloader(self.test_dataset, self.batch_size, num_workers=self.num_workers, collate_fn=collate_pocket)

    def forward(self, data, training):
        x = data['positions'].to(self.device) # position of scaffold+extension
        h = data['one_hot'].to(self.device) # onehot of scaffold+extension
        
        B = x.shape[0] # batch size
        N = x.shape[1] # number of atoms in scaffold+extension

        node_masks = data['atom_mask'].to(self.device) # masks for ligand atoms
        extension_masks = data['extension_masks'].to(self.device) # mask on only extension atoms
        scaffold_masks = data['scaffold_masks'].to(self.device) # mask on only scaffold atoms
        anchors = data['anchors'].to(self.device)
        pocket_anchors = data['pocket_anchors'].to(self.device)
        pocket_x = data['pocket_coords'].to(self.device)
        pocket_h = data['pocket_onehot'].to(self.device)
        pocket_masks = data['pocket_masks'].to(self.device)

        if self.data_augmentation:
            x = utils.random_rotation(x)

        # Remove COM of fragment from the molecule
        # NOTE: remove_partial_mean_with_mask first computes the com based on the last argument, then subtract it from the atoms in the second argument
        if self.center_of_mass == 'fragments':
            x = utils.remove_partial_mean_with_mask(x, node_masks, extension_masks.unsqueeze(-1)) # remove COM of next fragment
        
        elif self.center_of_mass == 'anchors':
            anchor_pos = torch.zeros((B,3), device=x.device)
            row1, col1 = torch.where(pocket_anchors)
            anchor_pos[row1] = pocket_x[row1, col1]

            row2, col2 = torch.where(anchors)
            anchor_pos[row2] = x[row2, col2]
            x = x - anchor_pos.unsqueeze(1) * node_masks.unsqueeze(-1)
            pocket_x = pocket_x - anchor_pos.unsqueeze(1) * pocket_masks.unsqueeze(-1)

            output = self.edm.forward(
                    x=x, # position of scaffold + extension
                    h=h, # onehot of  scaffold + extension
                    pocket_x=pocket_x, # position of pocket atoms (centered at anchor point)
                    pocket_h=pocket_h,
                    anchors=anchors,
                    pocket_anchors=pocket_anchors,
                    scaffold_mask=scaffold_masks, # masking on the scaffold + pocket atoms [B, N_l]
                    extension_mask=extension_masks, # masking on only the fragment atoms # [B, N_l]
                    pocket_mask=pocket_masks)
        else:
            raise ValueError
        return output
    
    def training_step(self, data, *args):
        output =  self.forward(data, training=True)  
        if output is None:
            return
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = output
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px

        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        training_metrics = {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

        #if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
        for metric_name, metric in training_metrics.items():
            #self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
            self.log(f'{metric_name}/train', metric, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
    
        torch.cuda.empty_cache()
        return training_metrics
    
    def validation_step(self, data, *args):

        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False) 
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px

        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        
        validation_metrics = {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

        return validation_metrics
        
    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

        torch.cuda.empty_cache()
        self.metrics.clear() # free up memory

    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)

        if (self.current_epoch + 1) % self.test_epochs == 0:

            sampling_results = self.sample_and_analyze()
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/val', metric_value, prog_bar=True)
                self.metrics.setdefault(f'{metric_name}/val', []).append(metric_value)

            # Logging the results corresponding to the best validation_and_connectivity
            best_metrics, best_epoch = self.compute_best_validation_metrics()
            self.log('best_epoch', int(best_epoch), prog_bar=True, batch_size=self.batch_size)
            for metric, value in best_metrics.items():
                self.log(f'best_{metric}', value, prog_bar=True, batch_size=self.batch_size)
        
        self.metrics.clear() # free up memory

    def generate_animation(self):
        pass

    def sample_batch_molecules_pocket(self, n_samples=10, use_anchors=True, random_anchors=True, keep_frames=None):
        pass


    def sample_and_analyze_molecules_pocket(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)
        return optimizer

    #def optimizer_step(
    #    self,
    #    epoch,
    #    batch_idx,
    #    optimizer,
    #    optimizer_idx,
    #    optimizer_closure,
    #    on_tpu=False,
    #    using_native_amp=False,
    #    using_lbfgs=False,
    #):
    #    # Execute the optimizer_closure
    #    optimizer_closure()
    #
    #    # Check for NaN values in model parameters or gradients
    #    nan_in_params = any(torch.isnan(p).any() for p in self.parameters())
    #    nan_in_grads = any(torch.isnan(p.grad).any() for p in self.parameters() if p.grad is not None)
    #    loss_less_than_thress = self.loss < 0.7
    #    # If there are no NaN values proceed with the optimizer step
    #    if not (nan_in_params or nan_in_grads or loss_less_than_thress):
    #        super().optimizer_step(
    #            epoch,
    #            batch_idx,
    #            optimizer,
    #            optimizer_idx,
    #            optimizer_closure,
    #            on_tpu,
    #            using_native_amp,
    #            using_lbfgs,
    #        )
    #    else:
    #        print("Skipping gradient update due to NaN values detected or loss exceeding the threshold.")

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'validity_and_connectivity/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        } 
        return best_metrics, best_epoch
    
    @staticmethod
    def aggregate_metric(step_outputs, metric):
        # Filtering out None values
        valid_values = [out[metric] for out in step_outputs if out is not None]
        if not valid_values:  # if all values were None, return None
            return None
        return torch.tensor(valid_values).mean()
