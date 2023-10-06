import argparse
import os
import pwd
import sys
import datetime
from src.const import GEOM_NUMBER_OF_ATOM_TYPES, CROSSDOCK_NUMBER_OF_ATOMS
from src.lightning_anchor_gnn import AnchorGNN_pl
from src.utils import disable_rdkit_logging, Logger
from pytorch_lightning import Trainer, callbacks, loggers
from pytorch_lightning.loggers import TensorBoardLogger
import wandb

def find_last_checkpoint(checkpoints_dir):
    epoch2fname = [
        (int(fname.split('=')[1].split('.')[0]), fname)
        for fname in os.listdir(checkpoints_dir)
        if fname.endswith('.ckpt')
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(checkpoints_dir, latest_fname)

def main(args):
    run_name = args.exp_name
    experiment = run_name if args.resume is None else args.resume
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    os.makedirs(os.path.join(args.logs, 'general_logs', experiment), exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(args.logs, "general_logs", experiment, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(args.logs, "general_logs", experiment, f'log.log'), syspart=sys.stderr)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    samples_dir = os.path.join(args.logs, 'samples', experiment)

    TB_Logger = TensorBoardLogger('tb_logs', name=experiment)
    wandb_logger = loggers.WandbLogger(
        save_dir=args.logs,
        project='diffusion-anchor-pred',
        name=experiment,
        id=experiment,
        resume='must' if args.resume is not None else 'allow'
    )

    if args.gaussian_expansion is not None:
        gaussian_expansion = True
    else:
        gaussian_expansion = False

    if args.use_guidance:
        use_guidance = True
    else:
        use_guidance = False

    if args.guidance_feature == 'QED' or args.guidance_feature == 'SA':
        guidance_classes = 6
    elif args.guidance_feature == 'Vina':
        guidance_classes = 6
    else:
        raise ValueError

    # ---------------------------------------------------------
    lig_nf = 10 # atom types
    pocket_nf = 25 # node features (4) + AA type (20) + BB (1)
    #context_node_nf = 3 # context is (anchors + scaffold_masks + pocket_masks )
    joint_nf = 32
    
    anchor_predictor = AnchorGNN_pl(            
            lig_node_nf=lig_nf,
            pocket_node_nf=pocket_nf,
            joint_nf=joint_nf, # TODO: change this?
            n_dims=3,
            hidden_nf=args.nf,
            activation=args.activation,
            tanh=args.tanh,
            n_layers=args.n_layers,
            attention=args.attention,
            norm_constant=args.norm_constant,
            data_path=args.data,
            train_data_prefix=args.train_data_prefix,
            val_data_prefix=args.val_data_prefix,
            batch_size=args.batch_size,
            lr=args.lr,
            test_epochs=args.test_epochs,
            normalization_factor=args.normalization_factor,
            normalization=args.normalization,
            include_charges=False,
            samples_dir=None,
            train_dataframe_path='paths_train.csv',
            val_dataframe_path='paths_val.csv',  
            num_workers=0,
            dataset_type=args.dataset_type,
            use_guidance=use_guidance,
            guidance_classes=guidance_classes,
            guidance_feature=args.guidance_feature,
            gaussian_expansion=gaussian_expansion)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=experiment+'_{epoch:02}',
        monitor='loss/val',
        save_top_k=10 
    )

    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        accelerator='gpu',
        devices=[0,1],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        strategy='ddp',
        precision=16
    )

    if args.resume is None:
        last_checkpoint = None
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the last checkpoint {last_checkpoint}')
    print('Start training')
    trainer.fit(model=anchor_predictor, ckpt_path=last_checkpoint)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='anchor_predictor')
    p.add_argument('--data', action='store', type=str,  default="")
    p.add_argument('--train-dataframe-path', action='store', type=str, default='paths_train.csv')
    p.add_argument('--valid-dataframe-path', action='store', type=str, default='paths_val.csv')
    p.add_argument('--train_data_prefix', action='store', type=str, default='train_data')
    p.add_argument('--val_data_prefix', action='store', type=str,  default='val_data')
    p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
    p.add_argument('--logs', action='store', type=str, default='logs')
    p.add_argument('--device', action='store', type=str, default='cuda:1')
    p.add_argument('--trainer_params', type=dict, help='parameters with keywords of the lightning trainer')
    p.add_argument('--log_iterations', action='store', type=str, default=20)
    p.add_argument('--exp_name', type=str, default='test_1')

    p.add_argument('--n_epochs', type=int, default=400)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)

    p.add_argument('--activation', type=str, default='silu', help='activation function')
    p.add_argument('--n_layers', type=int, default=4,   help='number of layers')
    p.add_argument('--inv_sublayers', type=int, default=2, help='number of layers')
    p.add_argument('--nf', type=int, default=128,  help='number of layers')
    p.add_argument('--tanh', type=eval, default=False, help='use tanh in the coord_mlp')
    p.add_argument('--attention', type=eval, default=False, help='use attention in the EGNN')
    p.add_argument('--norm_constant', type=float, default=100, help='diff/(|diff| + norm_constant)')
    
    p.add_argument('--resume', type=str, default=None, help='')
    p.add_argument('--start_epoch', type=int, default=0, help='')
    p.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    p.add_argument('--test_epochs', type=int, default=100)
    p.add_argument('--aggregation_method', type=str, default='sum',help='"sum" or "mean"')
    p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')
    p.add_argument('--normalization_factor', type=float, default=100, help="Normalize the sum aggregation of EGNN")
    p.add_argument('--dataset-type', type=str, default='GEOM', help='dataset-type can be GEOM or CrossDock for now')
    
    p.add_argument('--gaussian-expansion', action='store_true', default=False, help='whether to use gaussian expansion of distances')
    p.add_argument('--use-guidance', action='store_true', default=False, help='whether to train anchor-predictor for a specific guidance feature')
    p.add_argument('--guidance-feature', type=str, default='QED', help='guidance feature for adding to anchor predictor')
    args = p.parse_args()
    main(args=args)