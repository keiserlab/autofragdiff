import numpy as np
import torch
import pytorch_lightning as pl

from torch.nn.functional import sigmoid
from src.datasets import HierCrossDockDataset, get_dataloader, collate_pocket_aux
from src.anchor_gnn import AnchorGNNPocket

from typing import Dict, List, Optional
from tqdm import tqdm
import os
import torch.nn as nn

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception('activation fn not found. add it here')

class MaskedBCEWithLogitsLoss(torch.nn.Module):
    """ masks the pocket atoms for anchor prediction loss calculation """
    def __init__(self):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target, mask=None, return_mean=False):
        masked_loss = self.loss(input, target)

        if mask is not None:
            masked_loss = masked_loss * mask.float()
        if return_mean:
            if mask is not None:
                return masked_loss.sum() / mask.sum().float()
            else:
                return masked_loss.mean()
        else:
            return masked_loss
        
class AnchorGNN_pl(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    def __init__(
            self,
            lig_node_nf,
            pocket_node_nf,
            joint_nf,
            n_dims,
            hidden_nf,
            activation,
            tanh,
            n_layers,
            attention,
            norm_constant,
            data_path,
            train_data_prefix,
            val_data_prefix,
            batch_size,
            lr,
            test_epochs,
            dataset_type,
            normalization_factor,
            gaussian_expansion=False,
            normalization=None,
            include_charges=False,
            samples_dir=None,
            train_dataframe_path='paths_train.csv',
            val_dataframe_path='paths_val.csv',  
            num_workers=0, 
            ):
        
        super(AnchorGNN_pl, self).__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.test_epochs = test_epochs
        self.samples_dir = samples_dir
        self.n_dims = n_dims
        self.num_classes = lig_node_nf - include_charges
        self.include_charges = include_charges
        self.train_dataframe_path = train_dataframe_path
        self.val_dataframe_path = val_dataframe_path
        self.num_workers = num_workers
        self.n_layers = n_layers
        self.attention = attention
        self.normalization_factor = normalization_factor
        
        self.joint_nf = joint_nf
        self.lig_node_nf = lig_node_nf
        self.pocket_node_nf = pocket_node_nf

        self.norm_constant = norm_constant
        self.tanh = tanh 
        self.dataset_type = dataset_type
        self.gaussian_expansion = gaussian_expansion
        #self.bce_loss = MaskedBCEWithLogitsLoss()

        if self.dataset_type == 'GEOM':
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        elif self.dataset_type == 'CrossDock':
            self.bce_loss = MaskedBCEWithLogitsLoss()

        if type(activation) is str:
            activation = get_activation(activation)
        
        self.anchor_predictor = AnchorGNNPocket(
                lig_nf=lig_node_nf,
                pocket_nf=pocket_node_nf,
                joint_nf=joint_nf,
                hidden_nf=hidden_nf,
                out_node_nf=hidden_nf,
                n_layers=4,
                normalization_factor=normalization_factor,
                normalization=normalization,
                attention=True,
                aggregation_method='sum',
                dist_cutoff=7,
                gaussian_expansion=gaussian_expansion,
                edge_cutoff_ligand=None,
                edge_cutoff_pocket=4.5,
                edge_cutoff_interaction=4.5
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
        return get_dataloader(self.train_dataset, self.batch_size, num_workers=self.num_workers, collate_fn=collate_pocket_aux, shuffle=True)
    
    def val_dataloader(self):
        return get_dataloader(self.val_dataset, self.batch_size, num_workers=self.num_workers, collate_fn=collate_pocket_aux)
    
    def test_dataloader(self):
        return get_dataloader(self.test_dataset, self.batch_size, num_workers=self.num_workers, collate_fn=collate_pocket_aux)

    def forward(self, data, training):

        scaff_x = data['position_aux'].to(self.device) # [B, Ns, 3]
        scaff_h = data['onehot_aux'].to(self.device) # [B, Ns, nf]
        scaffold_masks = data['scaffold_masks_aux'].to(self.device) # [B, Ns]
        pocket_masks = data['pocket_mask_aux'].to(self.device) # [B, Np]
        scaffold_anchors = data['anchors_aux'].to(self.device) # [B,Ns]
        pocket_x = data['pocket_coords_aux'].to(self.device)
        pocket_h = data['pocket_onehot_aux'].to(self.device) 

        B, N  = scaff_x.shape[0], scaff_x.shape[1]

        B = scaff_x.shape[0]
        N_s = scaff_x.shape[1]
        N_p = pocket_x.shape[1]
        N = N_s+N_p

        anchor_out = self.anchor_predictor.forward(mol_x=scaff_x, # [B, Ns, 3]
                                                    mol_h=scaff_h, # [B, Ns, nf]
                                                    pocket_x=pocket_x, # [B, Np, 3]
                                                    pocket_h=pocket_h, # [B, Np, hp]
                                                    node_mask=scaffold_masks, # [B, Np] # mask on both pocket and scaffold
                                                    pocket_mask=pocket_masks,
                                                    )  # [B, Np] masks only on pocket atoms) 

        anchor_loss = self.bce_loss(anchor_out.view(B*N_s, 1), scaffold_anchors.view(B*N_s, 1), scaffold_masks.view(B*N_s, 1), return_mean=True)
        #anchor_loss = anchor_loss[not_first_frag_mask].mean()
        return anchor_out, anchor_loss
    
    def training_step(self, data, *args):
        _, loss = self.forward(data, training=True)
        training_metrics = {
            'loss': loss
        }
        for metric_name, metric in training_metrics.items():
            self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
            self.log(f'{metric_name}/train', metric, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.metrics.clear()
        return training_metrics

    def validation_step(self, data, *args):
        _, loss = self.forward(data, training=False)
        validation_metrics = {
            'loss': loss 
        }
        return validation_metrics
    
    def training_epoch_end(self, training_step_outputs):
        for metric in training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)

        self.metrics.clear() # free up memory
    
    def validation_epoch_end(self, validation_step_outputs):
        for metric in validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)
        
        self.metrics.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.anchor_predictor.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
