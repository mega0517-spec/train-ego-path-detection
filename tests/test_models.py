import pytest
import torch

from src.nn.backbone import EfficientNetBackbone, ResNetBackbone
from src.nn.decoder import UNetDecoder
from src.nn.model import ClassificationNet, RegressionNet, SegmentationNet

pytestmark = pytest.mark.slow

BACKBONES = ["resnet18", "efficientnet-b0"]
INPUT_SHAPE = (3, 64, 64)


class TestBackbones:
    @pytest.mark.parametrize("version", ["18", "34"])
    def test_resnet_returns_the_requested_levels(self, version):
        backbone = ResNetBackbone(version, out_levels=(1, 2, 3, 4, 5))
        features = backbone(torch.rand(1, *INPUT_SHAPE))
        assert len(features) == 5
        assert backbone.reduction_factor == 32
        for i, feature in enumerate(features):
            assert feature.shape[1] == backbone.out_channels[i]
            assert feature.shape[2] == INPUT_SHAPE[1] // 2 ** (i + 1)

    def test_efficientnet_returns_the_requested_levels(self):
        backbone = EfficientNetBackbone("b0", out_levels=(1, 3, 4, 6, 8))
        features = backbone(torch.rand(1, *INPUT_SHAPE))
        assert len(features) == 5
        for i, feature in enumerate(features):
            assert feature.shape[1] == backbone.out_channels[i]
            assert feature.shape[2] == INPUT_SHAPE[1] // 2 ** (i + 1)

    def test_level_zero_returns_the_input_itself(self):
        backbone = ResNetBackbone("18", out_levels=(0, 5))
        features = backbone(torch.rand(1, *INPUT_SHAPE))
        assert backbone.out_channels[0] == 3
        assert features[0].shape == (1, *INPUT_SHAPE)

    def test_default_level_is_the_last_stage_only(self):
        backbone = ResNetBackbone("18")
        features = backbone(torch.rand(1, *INPUT_SHAPE))
        assert len(features) == 1
        assert features[0].shape[2] == INPUT_SHAPE[1] // 32

    @pytest.mark.parametrize(
        "cls, version", [(ResNetBackbone, "101"), (EfficientNetBackbone, "b7")]
    )
    def test_unsupported_version_raises(self, cls, version):
        with pytest.raises(NotImplementedError):
            cls(version)


class TestUNetDecoder:
    def test_upsamples_back_to_the_input_resolution(self):
        encoder = ResNetBackbone("18", out_levels=(1, 2, 3, 4, 5))
        decoder = UNetDecoder(encoder.out_channels, (16, 8, 4, 4, 4))
        output = decoder(encoder(torch.rand(1, *INPUT_SHAPE)))
        assert output.shape == (1, 4, INPUT_SHAPE[1], INPUT_SHAPE[2])


class TestClassificationNet:
    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_output_shape(self, backbone):
        model = ClassificationNet(
            backbone=backbone,
            input_shape=INPUT_SHAPE,
            anchors=8,
            classes=16,
            pool_channels=2,
            fc_hidden_size=8,
        )
        output = model(torch.rand(2, *INPUT_SHAPE))
        # one logit per (rail, anchor, class) with the background class included
        assert output.shape == (2, 8 * (16 + 1) * 2)

    def test_rejects_an_unknown_backbone(self):
        with pytest.raises(NotImplementedError):
            ClassificationNet("mobilenet", INPUT_SHAPE, 8, 16, 2, 8)

    def test_handles_input_shapes_that_are_not_multiples_of_the_reduction_factor(self):
        shape = (3, 100, 70)  # ceil(100/32) == 4, ceil(70/32) == 3
        model = ClassificationNet(
            backbone="resnet18",
            input_shape=shape,
            anchors=4,
            classes=8,
            pool_channels=2,
            fc_hidden_size=8,
        )
        assert model(torch.rand(1, *shape)).shape == (1, 4 * 9 * 2)


class TestRegressionNet:
    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_output_shape(self, backbone):
        model = RegressionNet(
            backbone=backbone,
            input_shape=INPUT_SHAPE,
            anchors=8,
            pool_channels=2,
            fc_hidden_size=8,
        )
        # one x per (rail, anchor), plus a single y-limit logit
        assert model(torch.rand(2, *INPUT_SHAPE)).shape == (2, 8 * 2 + 1)

    def test_rejects_an_unknown_backbone(self):
        with pytest.raises(NotImplementedError):
            RegressionNet("mobilenet", INPUT_SHAPE, 8, 2, 8)


class TestSegmentationNet:
    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_output_shape(self, backbone):
        model = SegmentationNet(backbone=backbone, decoder_channels=(16, 8, 4, 4, 4))
        output = model(torch.rand(2, *INPUT_SHAPE))
        assert output.shape == (2, 1, INPUT_SHAPE[1], INPUT_SHAPE[2])

    def test_rejects_an_unknown_backbone(self):
        with pytest.raises(NotImplementedError):
            SegmentationNet("mobilenet", (16, 8, 4, 4, 4))

    def test_outputs_raw_logits(self):
        model = SegmentationNet(backbone="resnet18", decoder_channels=(16, 8, 4, 4, 4))
        output = model(torch.rand(1, *INPUT_SHAPE))
        # the head has no activation, so the loss is free to apply its own sigmoid
        assert output.min() < 0 or output.max() > 1


class TestTrainability:
    @pytest.mark.parametrize(
        "factory, target_shape",
        [
            (lambda: RegressionNet("resnet18", INPUT_SHAPE, 4, 2, 8), (1, 9)),
            (lambda: SegmentationNet("resnet18", (16, 8, 4, 4, 4)), (1, 1, 64, 64)),
        ],
    )
    def test_gradients_reach_the_backbone(self, factory, target_shape):
        model = factory()
        output = model(torch.rand(1, *INPUT_SHAPE))
        output.sum().backward()
        first_parameter = next(model.parameters())
        assert first_parameter.grad is not None
        assert torch.isfinite(first_parameter.grad).all()

    def test_eval_mode_is_deterministic(self):
        model = RegressionNet("resnet18", INPUT_SHAPE, 4, 2, 8).eval()
        sample = torch.rand(1, *INPUT_SHAPE)
        with torch.inference_mode():
            torch.testing.assert_close(model(sample), model(sample))
