import torch
class OnlyKeep1stDimension:
    def __call__(self, mei: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        """Change all values to kwargs.value (default 0) for all dimensions other than the first."""
        value = kwargs.get('value', 0)
        mei = mei.clone()
        mei[:, 1:, ...] = value
        return mei