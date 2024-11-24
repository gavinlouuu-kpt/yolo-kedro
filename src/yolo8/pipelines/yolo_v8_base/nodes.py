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
from PIL import Image
import tempfile
import os
import pickle
import yaml

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
        Tuple[Dict[str, np.ndarray], List[np.ndarray]]: Dictionary mapping image numbers to binary masks,
        and a list of debug visualization arrays
    """
    try:
        # Use LabelParser to convert annotations to masks
        masks = LabelParser.parse_json(annotations_input)
        logger.info(f"Successfully converted {len(masks)} annotations to masks")
        logger.info(f"5 random keys: {random.sample(list(masks.keys()), 5)}")
        
        # Create debug visualizations
        debug_images = {}
        for idx, (img_num, mask) in enumerate(random.sample(list(masks.items()), 5)):
            fig = plt.figure()
            plt.imshow(mask)
            plt.title(f"Mask for image {img_num}")
            # Convert plot to numpy array
            fig.canvas.draw()
            debug_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            debug_img = debug_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # Convert numpy array to PIL Image
            debug_img = Image.fromarray(debug_img)
            debug_images[f"debug_{idx}"] = debug_img
            plt.close(fig)
            
        return masks, debug_images
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
        train = {k: yolo_dataset_input[k] for k in train_keys}
        val = {k: yolo_dataset_input[k] for k in val_keys}
        test = {k: yolo_dataset_input[k] for k in test_keys}
        
        
        # Log split sizes
        logger.info(f"Dataset split sizes:")
        logger.info(f"Training: {len(train)} samples")
        logger.info(f"Validation: {len(val)} samples")
        logger.info(f"Test: {len(test)} samples")
        
        return train, val, test
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        raise

def fine_tune_yolo_v8_seg(train_input, val_input, test_input, parameters: Dict[str, Any]):
    """Fine-tune YOLOv8 segmentation model on custom dataset."""
    try:
        # Initialize model
        model = YOLO(parameters['model_name'])
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(temp_dir, split, 'images'), exist_ok=True)
                os.makedirs(os.path.join(temp_dir, split, 'labels'), exist_ok=True)
            
            # Copy data to appropriate directories
            # Assuming train_input, val_input, test_input are dictionaries with 'image' and 'mask' keys
            for idx, img_data in enumerate(train_input.values()):
                img_path = os.path.join(temp_dir, 'train', 'images', f'{idx}.jpg')
                mask_path = os.path.join(temp_dir, 'train', 'labels', f'{idx}.txt')
                img_data['image'].save(img_path)  # Save image
                # Convert mask to YOLO format and save
                # ... (mask conversion code here)
            
            for idx, img_data in enumerate(val_input.values()):
                img_path = os.path.join(temp_dir, 'val', 'images', f'{idx}.jpg')
                mask_path = os.path.join(temp_dir, 'val', 'labels', f'{idx}.txt')
                img_data['image'].save(img_path)
                # Convert mask to YOLO format and save
                # ... (mask conversion code here)
            
            for idx, img_data in enumerate(test_input.values()):
                img_path = os.path.join(temp_dir, 'test', 'images', f'{idx}.jpg')
                mask_path = os.path.join(temp_dir, 'test', 'labels', f'{idx}.txt')
                img_data['image'].save(img_path)
                # Convert mask to YOLO format and save
                # ... (mask conversion code here)
            
            # Create data.yaml configuration file
            data_yaml = {
                'path': temp_dir,  # Root directory
                'train': os.path.join('train', 'images'),  # Training images path
                'val': os.path.join('val', 'images'),      # Validation images path
                'test': os.path.join('test', 'images'),    # Test images path
                'nc': parameters.get('num_classes', 1),
                'names': parameters.get('class_names', ['object'])
            }
            
            # Save data.yaml configuration
            data_yaml_path = os.path.join(temp_dir, 'data.yaml')
            with open(data_yaml_path, 'w') as f:
                yaml.safe_dump(data_yaml, f)
            
            logger.info(f"Dataset structure created at {temp_dir}")
            logger.info(f"Training images: {len(os.listdir(os.path.join(temp_dir, 'train', 'images')))}")
            logger.info(f"Validation images: {len(os.listdir(os.path.join(temp_dir, 'val', 'images')))}")
            logger.info(f"Test images: {len(os.listdir(os.path.join(temp_dir, 'test', 'images')))}")
            
            # Prepare training arguments
            train_args = {
                'data': data_yaml_path,  # Path to data.yaml
                'epochs': parameters.get('epochs', 100),
                'imgsz': parameters.get('imgsz', 640),
                'batch': parameters.get('batch_size', 16),
                'device': parameters.get('device', 'cuda'),
                'val': True,
                'save': True,
                'save_period': parameters.get('save_period', -1),
                'patience': parameters.get('patience', 50),
                'project': parameters.get('project', 'yolo_training'),
                'name': parameters.get('run_name', 'exp'),
            }

            # Train the model
            results = model.train(**train_args)
            
            logger.info(f"Training completed. Results: {results}")
            return model

    except Exception as e:
        logger.error(f"Error in fine-tuning YOLO model: {str(e)}")
        raise