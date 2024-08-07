import torch

from module.piq.feature_extractors import InceptionV3
from module.piq.utils import _validate_features


class BaseFeatureMetric(torch.nn.Module):
    r"""Base class for all metrics, which require computation of per image features.
     For example: FID, KID, MSID etc.
     """

    def __init__(self) -> None:
        super(BaseFeatureMetric, self).__init__()

    def forward(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        # Sanity check for input
        _validate_features(predicted_features, target_features)
        return self.compute_metric(predicted_features, target_features)

    @torch.no_grad()
    def compute_feats(
            self,
            loader: torch.utils.data.DataLoader,
            feature_extractor: torch.nn.Module = None,
            device: str = 'cuda') -> torch.Tensor:
        r"""Generate low-dimensional image desciptors

        Args:
            loader: Should return dict with key `images` in it
            feature_extractor: model used to generate image features, if None use `InceptionNetV3` model.
                Model should return a list with features from one of the network layers.
            out_features: size of `feature_extractor` output
            device: Device on which to compute inference of the model
        """

        if feature_extractor is None:
            print('WARNING: default feature extractor (InceptionNet V2) is used.')
            feature_extractor = InceptionV3()
        else:
            assert isinstance(feature_extractor, torch.nn.Module), \
                f"Feature extractor must be PyTorch module. Got {type(feature_extractor)}"
        feature_extractor.to(device)
        feature_extractor.eval()

        total_feats = []
        for batch in loader:
            images = batch['images']
            N = images.shape[0]
            images = images.float().to(device)

            # Get features
            features = feature_extractor(images)
            # TODO(jamil 26.03.20): Add support for more than one feature map
            assert len(features) == 1, \
                f"feature_encoder must return list with features from one layer. Got {len(features)}"
            total_feats.append(features[0].view(N, -1))

        return torch.cat(total_feats, dim=0)

    def compute_metric(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This function should be defined for each children class")
