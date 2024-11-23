"""
This is a boilerplate pipeline 'yolo_v8_base'
generated using Kedro 0.19.9
"""
from typing import Dict, Any
import logging
import random
import numpy as np
from ultralytics import YOLO
from yolo8.utils.parse_label_json import LabelParser
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

# load lazy loading partitioned data to memory
def load_partitioned_data(partitioned_input):
    """Load all partitions of a partitioned dataset into memory.
    
    Args:
        partitioned_input: Either a dictionary with partition IDs as keys and load functions 
                          as values (for PartitionedDataset), or the actual dataset if not partitioned.
    
    Returns:
        list: List containing all loaded partition data, or single dataset if not partitioned
    """
    try:
        if isinstance(partitioned_input, dict):
            # Handle PartitionedDataset case
            loaded_data = []
            partition_items = list(sorted(partitioned_input.items()))
            
            # Sample 5 random partitions for shape logging
            sample_partitions = random.sample(
                partition_items, 
                min(5, len(partition_items))
            )
            
            # Log shapes of sampled partitions
            for partition_id, partition_load_func in sample_partitions:
                try:
                    img = partition_load_func()
                    # Convert PIL Image to numpy array to get shape
                    img_array = np.array(img)
                    logger.info(
                        f"Partition {partition_id} shape: {img_array.shape} (height, width, channels)"
                    )
                except Exception as e:
                    logger.error(f"Error loading sample partition {partition_id}: {str(e)}")
            
            # Load all partitions
            for partition_id, partition_load_func in partition_items:
                try:
                    data = partition_load_func()
                    loaded_data.append(data)
                except Exception as e:
                    logger.error(f"Error loading partition {partition_id}: {str(e)}")
                    continue
                    
            logger.info(f"Successfully loaded {len(loaded_data)} partitions")
            return loaded_data
        else:
            # Handle regular dataset case
            return partitioned_input
    except Exception as e:
        logger.error(f"Error in load_partitioned_data: {str(e)}")
        raise

def convert_rle_to_mask(annotations_input):
    """Convert RLE annotations from Label Studio format to binary masks.
    
    Args:
        annotations_input: List of annotation dictionaries in Label Studio JSON format
    
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping image numbers to binary masks
    """
    try:
        # Use LabelParser to convert annotations to masks
        masks = LabelParser.parse_json(annotations_input)
        logger.info(f"Successfully converted {len(masks)} annotations to masks")
        # log 5 random keys
        logger.info(f"5 random keys: {random.sample(list(masks.keys()), 5)}")
        # # debug plot 5 random masks
        # for img_num, mask in random.sample(list(masks.items()), 5):
        #     plt.imshow(mask)
        #     plt.title(f"Mask for image {img_num}")
        #     plt.show()
        return masks
    except Exception as e:
        logger.error(f"Error converting annotations to masks: {str(e)}")
        raise

# use the keys from masks and loaded images to create a yolo dataset
def create_yolo_dataset(masks_input, loaded_images_input):
    """Create a paired dataset of images and their corresponding masks.
    
    Args:
        masks_input (Dict[str, np.ndarray]): Dictionary of image IDs to binary masks
        loaded_images_input (List): List of loaded images from partitioned dataset
    
    Returns:
        Dict[str, Dict]: Dictionary containing paired images and masks
        Format: {
            'image_x': {
                'image': image_array,
                'mask': mask_array
            }
        }
    """
    try:
        dataset = {}
        # Create a dictionary of loaded images keyed by image ID
        images_dict = {
            f"image_{i}": img 
            for i, img in enumerate(loaded_images_input)
        }
        
        # Match masks with corresponding images
        for image_id, mask in masks_input.items():
            if image_id in images_dict:
                dataset[image_id] = {
                    'image': images_dict[image_id],
                    'mask': mask
                }
        
        logger.info(f"Created dataset with {len(dataset)} paired images and masks")
        # Log a few sample pairs to verify matching
        sample_keys = random.sample(list(dataset.keys()), min(3, len(dataset)))
        logger.info(f"Sample paired keys: {sample_keys}")
        
        return dataset
    
    except Exception as e:
        logger.error(f"Error creating YOLO dataset: {str(e)}")
        raise

# split the dataset into train, val, test
def split_dataset(yolo_dataset_input, parameters: Dict[str, Any]):
    """Split dataset into training, validation and test sets.
    
    Args:
        yolo_dataset_input (Dict[str, Dict]): Dictionary containing paired images and masks
        train_ratio (float): Proportion of data for training (default: 0.7)
        val_ratio (float): Proportion of data for validation (default: 0.15)
        test_ratio (float): Proportion of data for testing (default: 0.15)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        Dict[str, Dict]: Dictionary containing train, val, and test datasets
        Format: {
            'train': {image_id: {'image': image_array, 'mask': mask_array}, ...},
            'val': {...},
            'test': {...}
        }
    """
    train_ratio = parameters['train_ratio']
    val_ratio = parameters['val_ratio']
    test_ratio = parameters['test_ratio']
    random_state = parameters['random_state']

    try:
        # Verify split ratios sum to 1
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1")

        # Set random seed for reproducibility
        random.seed(random_state)
        
        # Get all keys and shuffle them
        all_keys = list(yolo_dataset_input.keys())
        random.shuffle(all_keys)
        
        # Calculate split indices
        n_samples = len(all_keys)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split keys
        train_keys = all_keys[:train_end]
        val_keys = all_keys[train_end:val_end]
        test_keys = all_keys[val_end:]
        
        # Create split datasets
        split_datasets = {
            'train': {k: yolo_dataset_input[k] for k in train_keys},
            'val': {k: yolo_dataset_input[k] for k in val_keys},
            'test': {k: yolo_dataset_input[k] for k in test_keys}
        }
        
        # Log split sizes
        logger.info(f"Dataset split sizes:")
        logger.info(f"Training: {len(split_datasets['train'])} samples")
        logger.info(f"Validation: {len(split_datasets['val'])} samples")
        logger.info(f"Test: {len(split_datasets['test'])} samples")
        
        return split_datasets
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        raise

# fine tune yolo v8 seg model with the dataset
def fine_tune_yolo_v8_seg(yolo_dataset_input, parameters: Dict[str, Any]):
    model = YOLO(parameters['model_name'])
    return model