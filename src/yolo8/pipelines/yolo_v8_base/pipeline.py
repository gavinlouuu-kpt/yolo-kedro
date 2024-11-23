"""
This is a boilerplate pipeline 'yolo_v8_base'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_partitioned_data, convert_rle_to_mask, create_yolo_dataset, split_dataset, fine_tune_yolo_v8_seg


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            load_partitioned_data,
            inputs="paa_12_1",
            outputs="paa_12_1_loaded",
        ),
        node(
            convert_rle_to_mask,
            inputs="paa_12_1_annotations",
            outputs="masks",
        ),
        node(
            create_yolo_dataset,
            inputs=["masks", "paa_12_1_loaded"],
            outputs="yolo_dataset",
        ),
        node(
            split_dataset,
            inputs=["yolo_dataset", "params:yolo_v8_fine_tune"],
            outputs="split_dataset",
        ),
        node(
            fine_tune_yolo_v8_seg,
            inputs=["split_dataset", "params:yolo_v8_fine_tune"],
            outputs="fine_tuned_model",
        )
    ])
