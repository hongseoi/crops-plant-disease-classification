import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import io
import logging
from sagemaker_inference import content_types, encoder, errors, utils


INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
DEFAULT_MODEL_FILENAME = "model_v1_scripted.pth"

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModelLoadError(Exception):
    pass

def _is_model_file(filename):
    is_model_file = False
    if os.path.isfile(filename):
        _, ext = os.path.splitext(filename)
        is_model_file = ext in [".pt", ".pth"]
    return is_model_file

def model_fn(model_dir):
    """Loads a model. For PyTorch, a default function to load a model only if Elastic Inference is used.
    In other cases, users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A PyTorch model.
    """
    if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Failed to load model with default model_fn: missing file {}.".format(
                    DEFAULT_MODEL_FILENAME
                )
            )
        # Client-framework is CPU only. But model will run in Elastic Inference server with CUDA.
        try:
            return torch.jit.load(model_path, map_location=torch.device("cpu"))
        except RuntimeError as e:
            raise ModelLoadError(
                "Failed to load {}. Please ensure model is saved using torchscript.".format(
                    model_path
                )
            ) from e
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            model_files = [
                file for file in os.listdir(model_dir) if _is_model_file(file)
            ]
            if len(model_files) != 1:
                raise ValueError(
                    "Exactly one .pth or .pt file is required for PyTorch models: {}".format(
                        model_files
                    )
                )
            model_path = os.path.join(model_dir, model_files[0])
        try:
            model = torch.jit.load(model_path, map_location=device)
        except RuntimeError as e:
            raise ModelLoadError(
                "Failed to load {}. Please ensure model is saved using torchscript.".format(
                    model_path
                )
            ) from e
        model = model.to(device)
        return model
    
def input_fn(input_data, content_type):
    # 요청 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize(size=128),
        transforms.ToTensor(),
    ])
    if content_type != "application/x-image":
        raise ValueError(f"type[{content_type}] not supported.")
    else:
        img = Image.open(input_data)
        img = transform(img)
        img = img[:, :128, :]
        img = torch.unsqueeze(img, 0)   
        return img

def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn
    Returns: a prediction
    """
    with torch.no_grad():
        if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
            device = torch.device("cpu")
            model = model.to(device)
            input_data = data.to(device)
            model.eval()
            with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                output = model([input_data])
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            input_data = data.to(device)
            model.eval()
            output = model([input_data])

    return output


def output_fn(prediction, accept):
    """A modified output_fn for PyTorch. Converts numeric predictions to text labels and serializes them.
    Args:
        prediction: a prediction result from predict_fn (numeric predictions)
        accept: type which the output data needs to be serialized
    Returns: output data serialized
    """
    if type(prediction) == torch.Tensor:
        # Convert numeric predictions to text labels (replace this with your label mapping logic)
        text_labels = convert_numeric_to_text_labels(prediction)
        # Now, prediction is a list of text labels

    for content_type in utils.parse_accept(accept):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            # Encode text labels
            encoded_prediction = encoder.encode(text_labels, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    raise errors.UnsupportedFormatError(accept)

def convert_numeric_to_text_labels(prediction):
    # Replace this with your label mapping logic
    # This is a placeholder example assuming you have a list of labels
    label_mapping = ['bean__bean_spot', 'bean__blight', 'bean__brown_spot', 'bean__healthy', 'corn__common_rust', 'corn__gray_spot', 'corn__healthy', 'green_onion__black_spot', 'green_onion__downy_mildew', 'green_onion__healthy', 'green_onion__rust', 'lectuce__downy_mildew', 'lectuce__drop', 'lectuce__healthy', 'pepper__anthracnose', 'pepper__healthy', 'pepper__powdery_mildew', 'potato__Early_Blight', 'potato__healthy', 'potato__late_Blight', 'potato__soft_rot', 'pumpkin__healthy', 'pumpkin__leaf_mold', 'pumpkin__mosaic', 'pumpkin__powdery_mildew', 'radish__black_spot', 'radish__downy_mildew', 'radish__healthy']

    _, preds = torch.max(prediction, dim=1)
    text_labels = [label_mapping[p] for p in preds]
    return text_labels


