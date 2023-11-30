import torch
import torch.nn.functional as F
import numpy as np
import math

from src import utils
from src.noise import GammaNetwork, PredefinedNoiseSchedule
from src.dynamics_gvp import DynamicsWithPockets
#from torch_scatter import scatter_mean

class AREDM(torch.nn.Module):
    def __init__(
            self,
            dynamics: DynamicsWithPockets,
            lig_nf: int, # number of lig features (10)
            pocket_nf: int, # number of pocket features (25)
            n_dims: int, # number of position features
            timesteps: int=500, 
            n_hier_steps=8, # 
            noise_schedule='learned',
            noise_precision=1e-5,
            loss_type='l2',
            norm_values=(1., 4., 1.),
            norm_biases=(None, 0., 0.),
            anchors_context=True, #
            center_of_mass='anchors',
    ):
        super().__init__()
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with vlb objective'
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)
        
        self.dynamics = dynamics
        self.lig_nf = lig_nf
        self.pocket_nf = pocket_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.hier_steps = n_hier_steps
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.anchors_context = anchors_context
        self.center_of_mass = center_of_mass

    def forward(self, x, h, pocket_x, pocket_h, extension_mask, scaffold_mask, anchors, pocket_anchors, pocket_mask):
        """ 
        inputs:
            x: [B, N_l, 3] coordinates of molecules
            h: [B, N_l, lig_nf] features of moleucles
            extension_masks: [B, N_l] mask on extension atoms
            scaffold_masks: [B, N_l] mask on scaffold atoms
            anchors: [B, N_l] mask on anchor atoms
            pocket_x: [B, N_p, 3] coordinates of pocket
            pocket_h: [B, N_p, pocket_nf] features of pocket atoms
            pocket_mask: [B, N_p] masking on all the pocket atoms
            pocket_anchors: [B, N_p] masking on anchor atoms
        """
        # NOTE: the molecule is already at the center of mass of the extension at the current step
        x, h = self.normalize(x, h)
        pocket_x, pocket_h = self.normalize(pocket_x, pocket_h)

        num_nodes = x.shape[1] + pocket_x.shape[1]
        # volume change loss term
        delta_log_px = self.delta_log_px(num_nodes) 
        # sample t
        t_int = torch.randint(0, self.T+1, size=(x.size(0), 1), device=x.device).float() # [B, 1]
        s_int = t_int - 1
        t = t_int / self.T
        s = s_int / self.T

        # masks for t=0 and t>0
        t_is_zero = (t_int==0).squeeze().float()
        t_is_not_zero = 1 - t_is_zero

        # compute gamma_t and gamma_s according to noise schedule
        gamma_t = self.inflate_batch_array(self.gamma(t), x) 
        gamma_s = self.inflate_batch_array(self.gamma(s), x)

        xh = torch.cat([x, h], dim=2) # [B, N_l, h_l+3]
        pocket_xh = torch.cat([pocket_x, pocket_h], dim=2) # [B, N_p, h_p+3]

        # compute alpha_t, sigma_t from gamma
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # sample noise (only for extension part)
        eps_t = self.sample_combined_position_features_noise(n_samples=x.size(0), n_nodes=x.size(1), mask=extension_mask.unsqueeze(-1))
        # keep the scaffold part unchanged
        z_t = alpha_t * xh + sigma_t * eps_t
        z_t = xh * scaffold_mask.unsqueeze(-1) + z_t * extension_mask.unsqueeze(-1) 
    
        eps_t_hat = self.dynamics.forward(
            xh=z_t, # zt has random at extension masks and real pos/feat at scaffold mask 
            t=t, # timestep
            pocket_xh=pocket_xh, 
            extension_mask=extension_mask, 
            scaffold_mask=scaffold_mask, 
            anchors=anchors,
            pocket_anchors=pocket_anchors, 
            pocket_mask=pocket_mask, # [B, N_p]
        ) # [B, N_l, lig_nf+3] # prediction only for the extension part of ligand

        eps_t_hat = eps_t_hat * extension_mask.unsqueeze(-1) # prediction only for the extension part # [B, N, nf+3]
        # computing basic error (further used for computing NLL and L2 loss)
        error_t = (eps_t - eps_t_hat) ** 2 # [B, N, nf+3]
        error_t = self.sum_except_batch(error_t)

        normalization = (self.n_dims + self.lig_nf) * self.numbers_of_nodes(extension_mask) 
        l2_loss = error_t / normalization
        l2_loss = l2_loss.mean()
        
        # KL between q(z_t|x) and p(z_T) = Normal(0,1)
        kl_prior = self.kl_prior(xh, num_nodes).mean()

        # computing NLL middle term
        SNR_weight = (self.SNR(gamma_s - gamma_t)-1).squeeze(1).squeeze(1)
        loss_term_t = self.T * 0.5 * SNR_weight * error_t
        loss_term_t = (loss_term_t * t_is_not_zero).sum() / t_is_not_zero.sum()

        # computing noise returned by dynamics
        noise = torch.norm(eps_t_hat, dim=[1,2])
        noise_t = (noise * t_is_not_zero).sum() / t_is_not_zero.sum()

        if t_is_zero.sum() > 0:
            # the _constants depending on sigma_0 from the cross ent term E_q(z0|x) [log p(x|z0)]
            neg_log_constants = -self.log_constant_of_p_x_given_z0(x, num_nodes)
            # compute L0 term (even if gamma_t is not actually gamma_0)
            # and selected only relevant via masking
            loss_term_0 = -self.log_p_xh_given_z0_without_constants(h, z_t, gamma_t, eps_t, eps_t_hat, extension_mask.unsqueeze(-1))
            loss_term_0 = loss_term_0 + neg_log_constants
            loss_term_0 = (loss_term_0 * t_is_zero).sum() / t_is_zero.sum()

            # computing noise returned by dynamics 
            noise_0 = (noise * t_is_zero).sum() / t_is_zero.sum()
        else:
            loss_term_0 = 0.
            noise_0 = 0.
        
        return delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0

    @torch.no_grad()
    def sample_chain_single_fragment(self, x, h, extension_mask, scaffold_mask, anchors, pocket_x, pocket_h, pocket_mask, pocket_anchors, lj_guidance=False, prot_mol_lj_rm=None, all_H_coords=None, guidance_weights=None, keep_frames=None):

        n_samples = x.size(0)
        n_nodes = x.size(1)

        x, h = self.normalize(x, h)
        pocket_x, pocket_h = self.normalize(pocket_x, pocket_h)

        xh = torch.cat([x,h], dim=2)
        pocket_xh = torch.cat([pocket_x, pocket_h], dim=2)
        
        # sampling from N(0,1)
        z = self.sample_combined_position_features_noise(n_samples, n_nodes, mask=extension_mask.unsqueeze(-1))

        z = xh * scaffold_mask.unsqueeze(-1) + z * extension_mask.unsqueeze(-1)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device) # [num_frames, num_samples, n_nodes, nf+3]
        # sample p(zs|zt)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples,1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt_only_extension(
                s=s_array,
                t=t_array,
                z_t=z,
                pocket_xh=pocket_xh,
                extension_mask=extension_mask,
                scaffold_mask=scaffold_mask,
                anchors=anchors,
                pocket_anchors=pocket_anchors,
                pocket_mask=pocket_mask,
            )

            # Move back to center of mass again
            #if self.center_of_mass == 'anchors':
            #    anchor_pos = torch.zeros((n_samples,3), device=x.device)
            #    row1, col1 = torch.where(pocket_anchors)
            #    anchor_pos[row1] = pocket_x[row1, col1]

            #    row2, col2 = torch.where(anchors)
            #    anchor_pos[row2] = x[row2, col2]
            #    z[:,:,:3] = z[:,:,:3] - anchor_pos.unsqueeze(1) * node_masks.unsqueeze(-1)
            #    pocket_xh[:,:,:3] = pocket_x[:,:,:3] - anchor_pos.unsqueeze(1) * pocket_mask.unsqueeze(-1)
            
            # NOTE: we are not doing LJ guidance in the last 20 steps
            if lj_guidance and s < 400:
                with torch.enable_grad():
                    lig_x, lig_h = z[:, :, :self.n_dims], z[:, :, self.n_dims:]
                    lig_x.requires_grad = True

                    #h_ext.requires_grad = True
                    if extension_mask.sum() != 0:
                        lj_prot_lig = self.compute_lj(lig_x, lig_h, extension_mask, pocket_xh, pocket_mask, prot_mol_lj_rm=prot_mol_lj_rm, all_H_coords=all_H_coords)
                        lj_grad = torch.autograd.grad(lj_prot_lig, lig_x, retain_graph=True)[0] # 
                        energy_prot_mol_lj = 0.3 * lj_grad * extension_mask.unsqueeze(-1) #  
                        energy_total = energy_prot_mol_lj * guidance_weights[s]
                        z[:, :, :self.n_dims] = z[:, :, :self.n_dims] - energy_total 
    
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)
        
        # finally sample p(x, h | z0)
        x_out , h_out = self.sample_p_xh_given_z0_only_extension(
            z_0=z,
            pocket_xh=pocket_xh,
            extension_mask=extension_mask,
            scaffold_mask=scaffold_mask,
            anchors=anchors,
            pocket_anchors=pocket_anchors,
            pocket_mask=pocket_mask
        )

        chain[0] = torch.cat([x_out, h_out], dim=2)
        return x_out, h_out, chain
        
    def compute_lj(self, lig_x, lig_h, extension_mask, pocket_xh, pocket_mask, prot_mol_lj_rm, all_H_coords):
        """ compute the LJ between protein and ligand 
        lig_x: [B, N, 3]
        lig_h: [B, N, hf]
        """

        #lig_x.requires_grad = True
        B = extension_mask.shape[0]
        lj_prot_lig = torch.tensor(0., device=lig_x.device)

        n = 0
        for j in range(B):
            num_atoms = extension_mask[j].sum()
            if num_atoms == 0:
                continue # no extension atoms
            n += 1

            x = lig_x[j][extension_mask[j].bool()]  # [N_l, 3] # onlly extension atoms
            h = lig_h[j][extension_mask[j].bool()]  # [N_l, hf] # only extension atoms

            pocket_x = pocket_xh[j][pocket_mask[j].bool()][:, :self.n_dims] # [N_p, 3]
            pocket_h = pocket_xh[j][pocket_mask[j].bool()][:, self.n_dims:][:,:4] # [N_p, hf]
            h_coords = all_H_coords[j] # [N_p, 3]

            # --------------- compute the LJ between protein and ligand --------------
            dists = torch.cdist(x, pocket_x, p=2)
            dists = torch.where(dists<0.5, 0.5, dists)
            inds_lig = torch.argmax(h, dim=1) # [N_l]
            inds_pocket = torch.argmax(pocket_h, dim=1).long() # [N_p]

            rm = prot_mol_lj_rm[inds_lig][:, inds_pocket] # [N_l, N_p] 
            lj = ((rm / dists) ** 12 - (rm / dists) ** 6) # [N_l, N_p]

            lj[torch.isnan(lj)] = 0

            lj = torch.clamp(lj, min=0, max=1000) # [N_l, N_p]

            # -------------  compute the loss for h atoms ----------------
            dists_h = torch.cdist(x, h_coords, p=2)
            dists_h = torch.where(dists_h<0.5, 0.5, dists_h)
            inds_H = torch.ones(len(h_coords), device=x.device).long() * 10 # index of H is 10 in the table
            rm_h = prot_mol_lj_rm[inds_lig][:, inds_H]
            lj_h = ((rm_h / dists_h) ** 12 - (rm_h / dists_h) ** 6) # [N_l, N_p]
            
            lj_h[torch.isnan(lj_h)] = 0 # remove nan values
            lj_h = torch.clamp(lj_h, min=0, max=1000) # [N_l, N_p]

            lj = lj.mean() 
            lj_h = lj_h.mean()

            lj_prot_lig = (lj_prot_lig + (lj + lj_h))
        
        lj_prot_lig = lj_prot_lig / n

        return lj_prot_lig

    @staticmethod
    def get_batch_mask(mask, device):
        n_nodes = mask.float().sum(dim=1).int()
        batch_size = mask.shape[0]
        batch_mask = torch.cat([torch.ones(n_nodes[i]) * i for i in range(batch_size)]).long().to(device)
        return batch_mask
    
    def sample_p_zs_given_zt_only_extension(self, s, t, z_t, pocket_xh, scaffold_mask, extension_mask, anchors, pocket_anchors, pocket_mask):
        """ samples zs ~ p(zs|zt) only used during sampling. Samples only the extension features and coords """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
        sigma_s = self.sigma(gamma_s, target_tensor=z_t)
        sigma_t = self.sigma(gamma_t, target_tensor=z_t)

        eps_hat = self.dynamics.forward(
            xh=z_t,
            pocket_xh=pocket_xh,
            t=t,
            extension_mask=extension_mask,
            scaffold_mask=scaffold_mask,
            anchors=anchors,
            pocket_anchors=pocket_anchors,
            pocket_mask=pocket_mask,
        )

        eps_hat = eps_hat * extension_mask.unsqueeze(-1) # only the extension part is unmasked
        # compute mu for p(z_s|z_t) (algorithm 2 EDM)
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat

        # compute sigma for p(z_s | z_t) (alg 2)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # sample z_s given the params for z_t
        z_s = self.sample_normal(mu, sigma, extension_mask.unsqueeze(-1))
        z_s = z_t * scaffold_mask.unsqueeze(-1) + z_s * extension_mask.unsqueeze(-1) # [B, N, 3]
        
        return z_s
    
    def sample_p_xh_given_z0_only_extension(self, z_0, pocket_xh, scaffold_mask, extension_mask, anchors, pocket_anchors, pocket_mask):

        zeros = torch.zeros(size=(z_0.size(0),1), device=z_0.device)
        gamma_0 = self.gamma(zeros)

        # compute sqrt(sigma_0^2/alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        eps_hat = self.dynamics.forward(
                                    t=zeros,
                                    xh=z_0,
                                    pocket_xh=pocket_xh,
                                    extension_mask=extension_mask,
                                    pocket_mask=pocket_mask,
                                    scaffold_mask=scaffold_mask,
                                    anchors=anchors,
                                    pocket_anchors= pocket_anchors,
                                    ) 

        eps_hat = eps_hat * extension_mask.unsqueeze(-1)
        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=extension_mask.unsqueeze(-1))
        xh = z_0 * scaffold_mask.unsqueeze(-1) + xh * extension_mask.unsqueeze(-1)

        x, h = xh[:, :, :self.n_dims], xh[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h)

        node_mask = (scaffold_mask.bool() | extension_mask.bool()).unsqueeze(-1).float()
        h = F.one_hot(torch.argmax(h, dim=2), self.lig_nf) * node_mask
        return x, h

    def kl_prior(self, xh, num_nodes):
        """
        computes the KL between q(z1|x) and prior p(z1) = Normal(0,1)
        This is essentially a lot of work for something that is in practice neglibible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule
        """
        # compute the last alpha value, alpha_T
        ones = torch.ones((xh.size(0),1), device=xh.device)
        gamma_T = self.gamma(ones) # the last step
        alpha_T = self.alpha(gamma_T, xh) # the last step

        # compute the means
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # compute standard devs (only batch axis for x-part, inflated for h-part)
        sigma_T_x = self.sigma(gamma_T, mu_T_x).view(-1) # remove inflate, only keep batch dim for x-part
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # compute KL for h-part
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = self.gaussian_kl(mu_T_h, sigma_T_h, zeros, ones)

        # compute KL for x-part
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_kl_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=d)

        return kl_distance_x + kl_distance_h

    def log_constant_of_p_x_given_z0(self, x, num_nodes):
        batch_size = x.size(0)
        degrees_of_freedom = self.subspace_dimensionality(num_nodes)
        zeros = torch.zeros((batch_size, 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)
        return degrees_of_freedom * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(self, h, z_0, gamma_0, eps, eps_hat, mask, epsilon=1e-10):
        """ compute log(p(x,h)|z0) without constants
        log p(x|z_0) (no const) = -0.5 ||eps-eps_hat||^2
        log p(h|z_0) (no const) = CDF(h + 0.5 - z_0)/sigma_0 - CDF(h - 0.5 - z_0)/sigma_0
            need to be normalized across categories -> log_sum_exp
        mask: must be the extension_mask for atoms to be generated
        """
        z_h = z_0[:, :, self.n_dims:]

        # take only part of over x
        eps_x = eps[:, :, :self.n_dims]
        eps_hat_x = eps_hat[:, :, :self.n_dims]

        # compute sigma_0 and rescale to integer scale of data
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0) * self.norm_values[1]

        # compute teh error for distribution N(x | 1/alpha_0 z_0 - sigma_0/alpha_0, sigma_0/alpha_0)
        # the weighting of eps parameterization is exactly 1 (eq 19 edm)
        # NOTE: for the next prediction hte previous parts of eps_x and eps_hat_x must be the same 
        log_p_x_given_z_without_constants = -0.5 * self.sum_except_batch((eps_x - eps_hat_x) ** 2)

        # categorical features
        # compute delta indicator mask
        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        # centered h_cat around 1, since onehot encoded
        centered_h = estimated_h - 1

        # compute integral from 0.5 to 1.5 of normal dist
        log_p_h_proportioal = torch.log(
            self.cdf_standard_gaussian(((centered_h + 0.5)/sigma_0)) -  # TODO: check if mask is actually needed here
            self.cdf_standard_gaussian(((centered_h - 0.5)/sigma_0)) + 
            epsilon
        )

        # normalize the distribution over categories
        log_Z = torch.logsumexp(log_p_h_proportioal, dim=2, keepdim=True)
        log_probabilities = log_p_h_proportioal - log_Z

        # select the log_prob of current category using onehot repr
        # mask is [B, N] boolean matrix
        log_p_h_given_z = self.sum_except_batch(log_probabilities * h * mask)

        # combine log probs for x and h
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z
        return log_p_xh_given_z

    def compute_x_pred(self, eps_t, z_t, gamma_t): # NOTE: may need a mask
        """ computes x_pred, i.e. the most likely prediction of x"""
        sigma_t = self.sigma(gamma_t, target_tensor=eps_t)
        alpha_t = self.alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        return x_pred

    def sample_combined_position_features_noise(self, n_samples, n_nodes, mask):
        """ samples gaussian noise for both positions and features and then concatenates them 
        uses node_mask on the noises
        """
        z_x = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=mask.device,
            node_mask=mask
        )
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.lig_nf),
            device=mask.device,
            node_mask=mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_normal(self, mu, sigma, node_mask):
        """ Samples from a normal dist """
        eps = self.sample_combined_position_features_noise(mu.size(0), mu.size(1), node_mask)
        return mu + sigma * eps
    
    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h
    
    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h 

    def unnormalize_z(self, z):
        assert z.size(2) == self.n_dims + self.lig_nf
        x, h  = z[:, :, :self.n_dims], z[:, :, self.n_dims:]
        x, h  = self.unnormalize(x, h)
        return torch.cat([x,h], dim=2)
    
    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * np.log(self.norm_values[0])

    def dimensionality(self, mask):
        return self.numbers_of_nodes(mask) * self.n_dims
    
    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)
    
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -self.expm1(self.softplus(gamma_s) - self.softplus(gamma_t)),
            target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def subspace_dimensionality(self, input_size):
        """Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined."""
        return (input_size - 1) * self.n_dims

    @staticmethod
    def numbers_of_nodes(mask):
        if len(mask.squeeze().shape) > 1:
            return torch.sum(mask.squeeze(), dim=1)
        else:
            return torch.sum(mask.squeeze())
    
    @staticmethod
    def inflate_batch_array(array, target):
         """ inflates teh batch array with only a single axis (i.e. shape = (batch_size,)
         or possibly more empty axes (batch_size, 1, ..., 1) to match the target)
         """
         target_shape = (array.size(0),) + (1,) * (len(target.size())-1)
         return array.view(target_shape)
    
    @staticmethod
    def sum_except_batch(x):
        return x.view(x.size(0), -1).sum(-1) # sum everyting except batch dimension

    @staticmethod
    def expm1(x):
        return torch.expm1(x)

    @staticmethod
    def softplus(x):
        return F.softplus(x)
    
    @staticmethod
    def cdf_standard_gaussian(x):
        """ Returns the cumulative distribution function (CDF) of gaussian """
        return 0.5 * (1 + torch.erf(x/math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu, q_sigma, p_mu, p_sigma):
        """
        Computes KL between two normal dists
        Args:
            q_mu: Mean of q
            q_sigma: std of q
            p_mu: mean of p
            p_sigma: std of p
        Returns:
            KL distance summed over all dims except the batch dim
        """
        kl = torch.log(p_sigma / q_sigma) + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) **2) / (p_sigma ** 2) - 0.5
        return AREDM.sum_except_batch(kl)

    def subspace_dimensionality(self, input_size):
        """Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined."""
        return (input_size - 1) * self.n_dims

    @staticmethod
    def gaussian_kl_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
        """ 
        computes the KL between two normal dists taking the dimension into account
        Returns:
            KL distance summed over all dims except the batch dim
        """
        mu_norm_2 = AREDM.sum_except_batch((q_mu - p_mu) ** 2)
        return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma ** 2 + mu_norm_2) / (p_sigma ** 2) - 0.5 * d