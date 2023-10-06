import argparse
import os
import pwd
import sys
import datetime

from src.const import GEOM_NUMBER_OF_ATOM_TYPES, CROSSDOCK_NUMBER_OF_ATOMS
from src.lightning import AR_DDPM
from src.utils import disable_rdkit_logging, Logger
from pytorch_lightning import Trainer, callbacks, loggers
from pytorch_lightning.loggers import TensorBoardLogger
import wandb
from src import const

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

    if args.dataset_type == 'GEOM':
        number_of_atoms = GEOM_NUMBER_OF_ATOM_TYPES
    elif args.dataset_type == 'CrossDock':
        number_of_atoms = CROSSDOCK_NUMBER_OF_ATOMS
    else:
        raise ValueError
    in_node_nf = number_of_atoms + args.include_charges
    anchors_context = not args.remove_anchors_context
    
    # ---------------------------------------------------------
    lig_nf = 10 # atom types (10) 
    pocket_nf = 25 # node features (4) + AA type (20) + BB (1)
    context_node_nf = 3 # context is (anchors + scaffold_masks + pocket_masks )
    TB_Logger = TensorBoardLogger('tb_logs', name=experiment)

    wandb_logger = loggers.WandbLogger(
        save_dir=args.logs,
        project='autofragdiff',
        name=experiment,
        id=experiment,
        resume='must' if args.resume is not None else 'allow',
    )
    
    torch_device = args.device
    joint_nf = 32 # 
    edge_cutoff_ligand = None
    edge_cutoff_pocket = 4.5
    edge_cutoff_interaction = 4.5

    ddpm = AR_DDPM(
        data_path=args.data,
        train_data_prefix=args.train_data_prefix,
        val_data_prefix=args.val_data_prefix,
        lig_nf=lig_nf,
        pocket_nf=pocket_nf,
        joint_nf=joint_nf,
        n_dims=3,
        context_node_nf=context_node_nf,
        hidden_nf=args.nf,
        activation=args.activation,
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        normalization_factor=args.normalization_factor,
        diffusion_steps=args.diffusion_steps,
        diffusion_noise_schedule=args.diffusion_noise_schedule,
        diffusion_noise_precision=args.diffusion_noise_precision,
        diffusion_loss_type=args.diffusion_loss_type,
        normalize_factors=args.normalize_factors,
        include_charges=args.include_charges,
        lr=args.lr,
        batch_size=args.batch_size,
        test_epochs=args.test_epochs,
        n_stability_samples=args.n_stability_samples,
        normalization=None,
        log_iterations=args.log_iterations,
        samples_dir=samples_dir,
        data_augmentation=args.data_augmentation,
        center_of_mass=args.center_of_mass,
        inpainting=args.inpainting,
        anchors_context=anchors_context,
        train_dataframe_path=args.train_dataframe_path,
        val_dataframe_path=args.valid_dataframe_path,
        num_workers=args.num_workers,
        dataset_type=args.dataset_type,
        gaussian_expansion=args.gaussian_expansion,
        num_gaussians=args.num_gaussians,
        edge_cutoff_ligand=edge_cutoff_ligand,
        edge_cutoff_pocket=edge_cutoff_pocket,
        edge_cutoff_interaction=edge_cutoff_interaction,
        clip_grad=False,
    )

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=experiment + '_{epoch:02d}',
        monitor='loss/val',
        save_top_k=20
    )

    trainer = Trainer(
        max_epochs=args.n_epochs,
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        accelerator='gpu',
        devices=[1,3],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        strategy='ddp',
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
    )

    if args.resume is None:
        last_checkpoint = None
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the last checkpoint {last_checkpoint}')
    print('Start training')
    trainer.fit(model=ddpm, ckpt_path=last_checkpoint)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='moldiffuser')
    #p.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
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
    p.add_argument('--model', type=str, default='egnn_dynamics',help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics |gnn_dynamics')
    p.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    p.add_argument('--diffusion_steps', type=int, default=500)
    p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
    p.add_argument('--diffusion_noise_precision', type=float, default=1e-5, )
    p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')

    p.add_argument('--n_epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--brute_force', type=eval, default=False, help='True | False')
    p.add_argument('--actnorm', type=eval, default=True,help='True | False')
    p.add_argument('--break_train_epoch', type=eval, default=False,help='True | False')
    p.add_argument('--dp', type=eval, default=True,help='True | False')
    p.add_argument('--condition_time', type=eval, default=True,help='True | False')
    p.add_argument('--clip_grad', type=eval, default=True, help='True | False')
    p.add_argument('--trace', type=str, default='hutch',help='hutch | exact')
    # EGNN args -->
    p.add_argument('--activation', type=str, default='silu', help='activation function')
    p.add_argument('--n_layers', type=int, default=6,  help='number of layers')
    p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
    p.add_argument('--nf', type=int, default=128,  help='number of layers')
    p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
    p.add_argument('--attention', type=eval, default=False, help='use attention in the EGNN')
    p.add_argument('--norm_constant', type=float, default=1., help='diff/(|diff| + norm_constant)')
    p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
    p.add_argument('--gaussian-expansion', action='store_true', default=False, help='whether to add gaussian expansion of distances')
    p.add_argument('--num-gaussians', type=int, default=16, help='number of gaussians for distances')
    p.add_argument('--ode_regularization', type=float, default=1e-3)
    p.add_argument('--dataset', type=str, default='qm9',  help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
    p.add_argument('--datadir', type=str, default='qm9/temp',  help='qm9 directory')
    p.add_argument('--filter_n_atoms', type=int, default=None, help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    p.add_argument('--dequantization', type=str, default='argmax_variational',  help='uniform | variational | argmax_variational | deterministic')
    p.add_argument('--n_report_steps', type=int, default=1)
    p.add_argument('--wandb_usr', type=str)
    p.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    p.add_argument('--enable_progress_bar', action='store_true', help='Disable wandb')
    p.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    p.add_argument('--no-cuda', action='store_true', default=False,  help='enables CUDA training')
    p.add_argument('--save_model', type=eval, default=True, help='save model')
    p.add_argument('--generate_epochs', type=int, default=1,help='save model')
    p.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    p.add_argument('--test_epochs', type=int, default=1000)
    p.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    p.add_argument("--conditioning", nargs='+', default=[], help='arguments : homo | lumo | alpha | gap | mu | Cv')
    p.add_argument('--resume', type=str, default=None, help='')
    p.add_argument('--start_epoch', type=int, default=0, help='')
    p.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    p.add_argument('--augment_noise', type=float, default=0)
    p.add_argument('--n_stability_samples', type=int, default=500,help='Number of samples to compute the stability')
    p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1], help='normalize factors for [x, categorical, integer]')
    p.add_argument('--remove_h', action='store_true')
    p.add_argument('--include_charges', type=eval, default=False,help='include atom charge or not') # TODO: change this
    p.add_argument('--visualize_every_batch', type=int, default=1e8,help="Can be used to visualize multiple times per epoch")
    p.add_argument('--normalization_factor', type=float, default=100,help="Normalize the sum aggregation of EGNN")
    p.add_argument('--aggregation_method', type=str, default='sum',help='"sum" or "mean"')
    p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')
    p.add_argument('--wandb_entity', type=str, default='geometric', help='Entity (project) name')
    p.add_argument('--center_of_mass', type=str, default='anchors', help='Where to center the data: fragments | anchors')
    p.add_argument('--inpainting', action='store_true', default=False, help='Inpainting mode (full generation)')
    p.add_argument('--remove_anchors_context', action='store_true', default=False, help='Remove anchors context')
    p.add_argument('--dataset-type', type=str, default='CrossDock', help='dataset type: GEOM, CrossDock')
   
    disable_rdkit_logging()

    args = p.parse_args()
    #if args.config:
    #    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    #    arg_dict = args.__dict__
    #    for key, value in config_dict.items():
    #        if isinstance(value, list) and key != 'normalize_factors':
    #            for v in value:
    #                arg_dict[key].append(v)
    #        else:
    #            arg_dict[key] = value
    #    args.config = args.config.name
    #else:
    #    config_dict = {}
    main(args=args)
