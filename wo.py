import os
import warnings

def turn_off_warnings():
    # Configure inference before importing it; the package reads these at import time.
    os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
    os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
    os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
    os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
    os.environ["DEPTH_ESTIMATION_ENABLED"] = "False"
    os.environ["INFERENCE_WARNINGS_DISABLED"] = "True"
    os.environ["IGNORE_MODEL_DEPENDENCIES_WARNINGS"] = "True"
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CPUExecutionProvider]"

    warnings.filterwarnings(
        "ignore",
        message=r".*DepthProImageProcessorFast.*deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*Importing from timm\.models\.layers is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Specified provider '.*ExecutionProvider' is not in available provider names.*",
    )