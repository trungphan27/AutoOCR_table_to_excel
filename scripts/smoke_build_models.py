"""Build all configured models on CPU without loading data or checkpoints."""

import copy
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLE_ROOT = PROJECT_ROOT / "training" / "PaddleOCR"
CONFIG_ROOT = PROJECT_ROOT / "training" / "configs"


def configure_output_channels(config, post_process):
    if not hasattr(post_process, "character"):
        return
    character_count = len(post_process.character)
    head = config["Architecture"]["Head"]
    if head["name"] == "MultiHead":
        head["out_channels_list"] = {
            "CTCLabelDecode": character_count,
            "SARLabelDecode": character_count + 2,
        }
        sar_loss = config["Loss"]["loss_config_list"][1]["SARLoss"]
        if sar_loss is None:
            config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                "ignore_index": character_count + 1
            }
        else:
            sar_loss["ignore_index"] = character_count + 1
    else:
        head["out_channels"] = character_count


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run this script with the project Python 3.12 venv.")
    if not PADDLE_ROOT.is_dir():
        raise FileNotFoundError(PADDLE_ROOT)

    sys.path.insert(0, str(PADDLE_ROOT))
    os.chdir(PADDLE_ROOT)

    import paddle
    import paddle.nn.layer.conv as paddle_conv

    paddle.set_device("cpu")
    # Paddle 2.6 GPU wheels query cuDNN while constructing Conv2D on CPU.
    # CUDA library validation is handled by check_training_env.py.
    paddle_conv.get_cudnn_version = lambda: None

    from ppocr.losses import build_loss
    from ppocr.metrics import build_metric
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from tools.program import load_config

    for component in ("det", "rec", "slanet"):
        config = copy.deepcopy(
            load_config(str(CONFIG_ROOT / "pubtabnet_{}.yml".format(component)))
        )
        config["Global"]["use_gpu"] = False
        backbone = config["Architecture"].get("Backbone")
        if isinstance(backbone, dict) and "pretrained" in backbone:
            backbone["pretrained"] = False
        post_process = build_post_process(config["PostProcess"], config["Global"])
        configure_output_channels(config, post_process)
        model = build_model(config["Architecture"])
        loss = build_loss(config["Loss"])
        metric = build_metric(config["Metric"])
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        print(
            "{}: model={}, loss={}, metric={}, parameters={:,}".format(
                component,
                type(model).__name__,
                type(loss).__name__,
                type(metric).__name__,
                parameter_count,
            )
        )


if __name__ == "__main__":
    main()
