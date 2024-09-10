from torch import nn
import logging
import torch

logger = logging.getLogger(__name__)

class ModulatedFiringRateEncoder(nn.Module):
    def __init__(self, core, readout, *, shifter=None, modulator=None, elu_offset=0.0):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model.
        The output is one positive value that can be interpreted as a firing rate, for example for a Poisson distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            elu_offset (float): Offset value in the final elu non-linearity. Defaults to 0.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            modulator (optional[nn.ModuleDict]): Modulator network. Modulator networks are now implemented. Defaults to None.
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = elu_offset

    def forward(
        self,
        inputs,
        *args,
        targets=None,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        history=None,
        state=None,
        rank_id=None,
        **kwargs
    ):
        x = self.core(inputs)
        if detach_core:
            x = x.detach()

        if self.shifter:
            if pupil_center is None:
                # raise ValueError("pupil_center is not given")
                # logger.warning("pupil_center is not given")
                shift = torch.tensor([0]).unsqueeze(0).to(x.device)
            else:
                shift = self.shifter[data_key](pupil_center, trial_idx)

        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)

        if self.modulator:
            # modulator contains non-linearities
            x = self.modulator[data_key](x, history=history,
                                        state=state, rank_id=rank_id)
        else:
            x = nn.functional.elu(x + self.offset) + 1
            
        return x

    def regularizer(self, data_key=None, reduction="sum", average=None, detach_core=False):
        reg = self.core.regularizer().detach() if detach_core else self.core.regularizer()
        reg += self.readout.regularizer(data_key=data_key, reduction=reduction, average=average)
        if self.shifter:
            reg += self.shifter.regularizer(data_key=data_key)
        if self.modulator:
            reg += self.modulator[data_key].regularizer()
        return reg



class FiringRateEncoder(nn.Module):
    def __init__(self, core, readout, *, shifter=None, modulator=None, elu_offset=0.0):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model.
        The output is one positive value that can be interpreted as a firing rate, for example for a Poisson distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            elu_offset (float): Offset value in the final elu non-linearity. Defaults to 0.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            modulator (optional[nn.ModuleDict]): Modulator network. Modulator networks are not implemented atm (24/06/2021). Defaults to None.
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = elu_offset

    def forward(
        self,
        inputs,
        *args,
        targets=None,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):
        x = self.core(inputs)
        if detach_core:
            x = x.detach()

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)

        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")
            x = self.modulator[data_key](x, behavior=behavior)

        return nn.functional.elu(x + self.offset) + 1

    def regularizer(self, data_key=None, reduction="sum", average=None, detach_core=False):
        reg = self.core.regularizer().detach() if detach_core else self.core.regularizer()
        reg += self.readout.regularizer(data_key=data_key, reduction=reduction, average=average)
        if self.shifter:
            reg += self.shifter.regularizer(data_key=data_key)
        if self.modulator:
            reg += self.modulator.regularizer(data_key=data_key)
        return reg
