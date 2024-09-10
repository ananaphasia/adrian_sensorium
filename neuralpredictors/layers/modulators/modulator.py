import torch
from torch import nn
from torch.nn import ModuleDict
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class HistoryStateGainModulator(nn.Module):
    def __init__(self, nr_neurons,
                 nr_trials,
                 include_history=True,
                 nr_history=5,
                 behav_state=False,
                 nr_behav_state=10,
                 include_gain=False,
                 gain_kernel_std=30,
                 diff_reg=1000,
                 per_neuron_gain_adjust=False,
                 gain_adjust_alpha=0,
                 alpha_behav=0,
                 alpha_hist=0,
                 ):
        
        super().__init__()
        # save parameter
        self.nr_trials = nr_trials
        self.include_history = include_history
        self.behav_state = behav_state
        self.include_gain = include_gain
        self.diff_reg = diff_reg
        self.per_neuron_gain_adjust = per_neuron_gain_adjust
        self.gain_adjust_alpha = gain_adjust_alpha
        self.alpha_behav = alpha_behav
        self.alpha_hist = alpha_hist
        
        if self.include_gain:
            max_val = np.sqrt( 1/nr_trials )
            weights = torch.rand( (nr_trials) ) * (2*max_val) - max_val
            self.own_gain = nn.Parameter( weights )
            # kernel is half a gaussian, to keep causal smoothing
            window = signal.gaussian(201, std=gain_kernel_std)
            window[0:100] = 0
            window = window / np.sum(window)   # normalize to area 1
            self.gain_kernel = torch.zeros( (1,1,201) )
            self.gain_kernel[0,0,:] = torch.Tensor(window)
            
            
        if self.per_neuron_gain_adjust:
            # initialize all close to zero
            max_val_g = np.sqrt( 1/nr_neurons )
            weights_g = torch.rand( (nr_neurons) ) * (2*max_val_g) - max_val_g
            self.gain_coupling = nn.Parameter( weights_g )
            self.coupling_offset = nn.Parameter( torch.zeros( 1 ) )
            # this value will be mapped to positive only values with elu+1
        
        if self.include_history:
            # initialize like linear layer, uniform between +-sqrt(1/nr)
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            max_val = np.sqrt( 1/nr_history )
            weights = torch.rand( (nr_neurons,nr_history) ) * (2*max_val) - max_val
            bias = torch.rand( nr_neurons ) * (2*max_val) - max_val
            self.history_weights = nn.parameter.Parameter( weights )
            self.history_bias = nn.parameter.Parameter( bias )
        
        if self.behav_state:
            # linear layer from hidden states to neurons
            self.state_encoder = nn.Linear( in_features=nr_behav_state,
                                            out_features=nr_neurons,
                                            bias=True )
            
            
    def forward(self, x, history=None, state=None, rank_id=None, device='cuda'):
        # x: (batch, nr_neurons) Output of the encoding model which uses images+behavior
        # history: (batch, nr_neurons, nr_lags)
        # gain: (batch, 1)
        # state: (batch, nr_states)
        # rank_id: (batch, 1)
        
        
        if self.include_history:
            # compute effect of history
            if history is None or self.history_bias is None:
                # logger.warning('History is not given or not initialized')
                hist = torch.zeros( (x.shape[0], x.shape[1]) ).to(device)
            else:
                hist = torch.einsum( 'bnh,nh->bn', history, self.history_weights )
            hist = hist + self.history_bias
            x = x + hist    # add history
            
        # add additional signal based on the behavioral state
        if self.behav_state:
            if state is None:
                # logger.warning('Behavioral state is not given')
                state_mod = torch.zeros( (x.shape[0], x.shape[1]) ).to(device)
            else:
                state_mod = self.state_encoder( state )
            x = x + nn.functional.elu( state_mod )
            

        # non-linearity for final output of stimulus segment, only positive values
        x = nn.functional.elu(x) + 1
        
        # modify stimulus response with gain
        if self.include_gain and rank_id is not None:
            # transform rank_ids into one-hot encoding
            nr_batch = x.shape[0]
            one_hot = torch.zeros( (nr_batch, 1, self.nr_trials)).to(device)
            for i, r_id in enumerate( rank_id[:,0] ):
                one_hot[i,0,int(r_id)] = 1
                
            # smooth one_hot encoding along trial dimension
            # one_hot has shape (batch, 1, nr_trials), interpreted as 1 channel
            # kernel has shape (1, 1, length), one input and one output channel
            self.gain_kernel = self.gain_kernel.to(device)
            smooth = nn.functional.conv1d(one_hot, self.gain_kernel, padding='same')

            # scalar product between smooth and saved gain vector (along trials)
            batch_gain = torch.einsum( 'bet,t->be', smooth, self.own_gain )

            # transform onto positive values only (0 mapped to 1)
            batch_gain = nn.functional.elu( batch_gain ) + 1
            
            if self.per_neuron_gain_adjust:
                # transform coupling values to 0 to pos values           # offset for all neurons
                coupling_value = nn.functional.elu( self.gain_coupling + self.coupling_offset ) + 1
                #                               (nr_neurons)    (batch,1)    
                adj_gain = nn.functional.elu( coupling_value * (batch_gain-1) ) + 1  # (batch,nr_neurons)
                
                x = adj_gain * x  
            else:
                x = batch_gain * x 

        return x
        
    def initialize(self, **kwargs):
        print('Initialize called but not implemented')
    

    def regularizer(self):
        """Add regularization for smooth own_gain and sparse gain_coupling """
        regularization = 0

        if self.include_gain and (self.diff_reg > 0):
            # L2 regularization of difference of gains
            smooth_gain = self.own_gain.diff().square().sum() * self.diff_reg
            regularization += smooth_gain

        if self.per_neuron_gain_adjust and (self.gain_adjust_alpha  > 0):
            sparse_gain_coupling = self.gain_coupling.abs().sum() * self.gain_adjust_alpha 
            regularization += sparse_gain_coupling
        
        if self.include_history and self.alpha_hist > 0:
            sum_hist = self.history_weights.abs().sum()
            regularization += sum_hist * self.alpha_hist
        
        if self.behav_state and self.alpha_behav > 0:
            sum_behav = self.state_encoder.weight.abs().sum()  # leave bias out
            regularization += sum_behav * self.alpha_behav
        
        return regularization

        
        
