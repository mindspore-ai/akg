import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs FeedsRepeat operation.
    """

    def __init__(self, output_feeds_size=500):
        super(Model, self).__init__()
        self.output_feeds_size = output_feeds_size

    def forward(self, feeds, feeds_repeat_times):
        """
        Perform FeedsRepeat operation.

        Args:
            feeds: Input feeds tensor
            feeds_repeat_times: Repeat times for each feed

        Returns:
            Repeated and padded feeds tensor
        """
        # Repeat feeds according to feeds_repeat_times
        repeated_feeds = torch.repeat_interleave(
            feeds, feeds_repeat_times, dim=0)

        # Calculate total repeated size
        total_repeated = torch.sum(feeds_repeat_times)

        # Calculate padding needed
        pad_size = self.output_feeds_size - total_repeated

        if pad_size > 0:
            # Pad with zeros to reach output_feeds_size
            pad_shape = list(repeated_feeds.shape)
            pad_shape[0] = pad_size
            padding = torch.zeros(
                pad_shape, dtype=repeated_feeds.dtype, device=repeated_feeds.device)
            output = torch.cat([repeated_feeds, padding], dim=0)
        else:
            # Truncate if output_feeds_size is smaller than total repeated
            output = repeated_feeds[:self.output_feeds_size]

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: feeds = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
    feeds = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    feeds_repeat_times = torch.tensor([100, 200], dtype=torch.int32)

    return [feeds, feeds_repeat_times]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [500]  # output_feeds_size=500
