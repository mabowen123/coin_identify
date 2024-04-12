import argparse
import os
import sys

sys.path.append("../")
import torch
import torch.nn as nn
import yaml
from timm.models import create_model


class ModelWithFlatten(nn.Module):
    def __init__(self, original_model):
        super(ModelWithFlatten, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = original_model.classifier

    def forward(self, x):
        flatten_output = self.features(x)
        # x = self.global_pool(x)
        # x = torch.flatten(x, 1)
        # flatten_output = self.flatten(x)
        classification_result = self.classifier(flatten_output)
        return flatten_output, classification_result


class CustomFC(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomFC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ResNet50FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            2048, 512
        )  # 输入通道为最后一个卷积层的输出通道数，输出为512维

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        flatten_output = self.fc(x)
        return flatten_output


class ModelWithFlattenRes(nn.Module):
    def __init__(self, original_model):
        super(ModelWithFlattenRes, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = original_model.fc

    def forward(self, x):
        x = self.features(x)
        # x = self.global_pool(x)
        # x = torch.flatten(x, 1)
        flatten_output = self.flatten(x)
        classification_result = self.classifier(flatten_output)
        return flatten_output, classification_result


class ModelWithFCS2(nn.Module):
    def __init__(self, original_model):
        super(ModelWithFCS2, self).__init__()
        self.features = original_model
        self.fc = nn.Linear(original_model.num_classes, 2016)

    def forward(self, x):
        x = self.features(x)
        flatten_output = torch.flatten(x, 1)
        fc_output = self.fc(x)
        return fc_output


def argument_parser() -> dict:
    """Argument Parser

    Returns
    -------
    config : dict
        Python dict containing all settings
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single_model",
        action="store_true",
        help="Flag to converts a single model \
                              (use full path + filename as \
                              --folder_path; input dimensions \
                              (--input_dim) and number of \
                              classes (--num_classes) are required)",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="model name (e.g. mobilenetv3_large_100)\
                            required if single_model",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path of folder containing training \
                              output (incl. yaml files)",
    )
    parser.add_argument(
        "--best_model_only",
        action="store_true",
        help="Flag to export model_best.pth.tar in \
                              training folder only",
    )
    parser.add_argument(
        "--last_model_only",
        action="store_true",
        help="Flag to export last.pth.tar in \
                              training folder only",
    )
    parser.add_argument(
        "--input_dim",
        default=None,
        type=int,
        help="Input shape of model assuming square\
                              input (example: --input_dim 224); \
                              if not provided, input dim is derived\
                              from folder name",
    )
    parser.add_argument(
        "--num_classes", type=int, default=None, help="Number of output classes"
    )
    args = parser.parse_args()
    config = {}
    config["single_model"] = args.single_model
    config["best_model_only"] = args.best_model_only
    config["last_model_only"] = args.last_model_only
    if config["best_model_only"] and config["last_model_only"]:
        raise Exception(
            "invalid options: best_model_only and\
            last_model_only - choose one"
        )
    config["folder_path"] = args.folder_path

    if not config["single_model"]:
        if config["folder_path"][-1] != "/":
            config["folder_path"] += "/"
        if not os.path.exists(config["folder_path"]):
            raise Exception(
                "Folder containing training output\
                does not exist"
            )
        if not os.path.exists(config["folder_path"] + "args.yaml"):
            raise Exception("args.yaml does not exist in folder")
        else:
            # derive data from folder name
            # training folder name format:
            # YYYYMMDD-hh:mm:ss-modelName_epochs-inputSize
            folder_name = config["folder_path"].split("/")[-2]
            timm_train_data = folder_name.split("-")
            config["input_dim"] = int(timm_train_data[3])
            config["timm_cfg"] = yaml.safe_load(
                open(config["folder_path"] + "args.yaml", "r")
            )
            config["timm_cfg_keys"] = tuple(config["timm_cfg"].keys())
            if "model" in config["timm_cfg_keys"]:
                config["model_name"] = config["timm_cfg"]["model"]
            else:
                config["model_name"] = timm_train_data[2]
            files_in_folder = os.listdir(config["folder_path"])
            config["models_in_folder"] = [
                file for file in files_in_folder if ".pth.tar" in file
            ]
            if len(config["models_in_folder"]) == 0:
                raise Exception("No models found in folder")
            if config["last_model_only"]:
                if "last.pth.tar" not in config["models_in_folder"]:
                    raise Exception("model last.pth.tar not found")
                else:
                    config["models_in_folder"] = ["last.pth.tar"]
            if config["best_model_only"]:
                if "model_best.pth.tar" not in config["models_in_folder"]:
                    raise Exception("model model_best.pth.tar not found")
                else:
                    config["models_in_folder"] = ["model_best.pth.tar"]
    else:
        if not os.path.exists(config["folder_path"]):
            raise Exception("Input model not found")
    config["num_classes"] = args.num_classes
    if not config["single_model"]:
        if config["num_classes"] is None:
            if "num_classes" in config["timm_cfg_keys"]:
                if config["timm_cfg"]["num_classes"] is None:
                    raise Exception("Number of classes required")
                else:
                    config["num_classes"] = config["timm_cfg"]["num_classes"]
    else:
        if config["num_classes"] is None:
            raise Exception("Number of classes required")
    if args.input_dim is None:
        if config["single_model"]:
            raise Exception("Input dimension missing")
    else:
        config["input_dim"] = args.input_dim
    if args.model_name is not None:
        config["model_name"] = args.model_name
    else:
        if config["single_model"]:
            raise Exception("Model name missing")
    return config


def parser_tensor(tensor, use="normal"):
    name = tensor.name
    print(f"{use} tensor name: {name}")

    data_type = tensor.type.tensor_type.elem_type
    print(f"{use} tensor data type: {data_type}")

    dims = tensor.type.tensor_type.shape.dim
    shape = []
    for i, dim in enumerate(dims):
        shape.append(dim.dim_value)
    print(f"{use} tensor with shape:{shape} ")


def convert_model(model_in: str, model_out: str, config: dict):
    random_input = torch.randn(
        1, 3, config["input_dim"], config["input_dim"], requires_grad=True
    )
    model = create_model(
        config["model_name"],
        checkpoint_path=model_in,
        exportable=True,
        num_classes=config["num_classes"],
    )
    model.eval()
    torch.onnx.export(
        model,
        random_input,
        model_out,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    import onnx

    onnx_model = onnx.load(model_out)
    graph = onnx_model.graph
    outputs = graph.output
    for output in outputs:
        parser_tensor(output, "output")


def main():
    config = argument_parser()
    print("=" * 72)
    print("Exporting models to ONNX")
    if not config["single_model"]:
        for model_file in config["models_in_folder"]:
            print(
                "Converting "
                + model_file
                + " to "
                + str(model_file.split(".")[0])
                + ".onnx"
            )
            model_in = config["folder_path"] + model_file
            model_out = model_in.replace("pth.tar", "onnx")
            convert_model(model_in, model_out, config)
    else:
        model_in = config["folder_path"]
        model_out = model_in.replace("pth.tar", "onnx")
        convert_model(model_in, model_out, config)
    print("=" * 72)


if __name__ == "__main__":
    main()
