import os
import sys
import time

import yaml
from tqdm import tqdm
from typing import Dict

import torch
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from monai.transforms import Activations
from monai.transforms import AsDiscrete
import nibabel as nib

sys.path.append("../")

from architectures.build_architecture import build_architecture

def load_config(config_path: str) -> Dict:
    """Loads the yaml config file

    Args:
        config_path (str): Path to the config file

    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_flair_raw_volume(case_name, brats_raw_root):
    """
    Load the complete Flair image from the original file and perform the same cropping as during training.
    """
    flair_path = os.path.join(brats_raw_root, "train", "imagesTr", f"{case_name}_0000.nii.gz")
    flair_full = nib.load(flair_path).get_fdata()  # (240, 240, 155)

    # Crop the central region corresponding to crop_brats2021_zero_pixels
    flair_crop = flair_full[56:184, 56:184, 13:141]  # -> (128, 128, 128)

    # Normalize to 0~1 (consistent with training). This use a MinMax Scaling.
    flair_crop = (flair_crop - flair_crop.min()) / (flair_crop.max() - flair_crop.min())

    return flair_crop.astype(np.float32)


def get_max_slice(vol):
    '''
    Get the slice with the largest area in a 3D volume
    -----------------------------------------------
    Parameters:
    - vol: 3D numpy array
    -----------------------------------------------
    Returns:
    - max_slice: int
    - max_area: int

    '''
    _, _, z = vol.shape
    max_area = 0
    max_slice = 0

    for i in range(z):
        area = np.sum(vol[:, :, i])
        if area > max_area:
            max_area = area
            max_slice = i

    return max_slice, max_area


def save_biggest_area_img(vol_og, vol_gt, vol_pred, max_slice, save_path):
    '''
    Save the image with the largest area in the 3D volume as a PNG file
    ------------------------------------------------
    Parameters:
    -----------------------------------------------
    - vol_og: 3D numpy array: Channel from the input tensor
    - vol_gt: 3D numpy array: Ground truth segmentation volume
    - vol_pred: 3D numpy array: Predicted segmentation volume
    - max_slice: int: Slice with the largest area in the 3D volume
    - max_area: int: Area of the largest slice
    - save_path: str: Path to save the image
    -----------------------------------------------
    Returns:
    - None

    '''
    # Define the colors for each label (0: background, 1: non-enhancing, 2: whole tumor, 3: enhancing tumor)
    colors = [(0, 0, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255)]  # RGB values: black, yellow, red, cyan
    # print(vol_og.shape, vol_gt.shape, vol_pred.shape)

    cmap = mcolors.ListedColormap(np.array(colors) / 255.0)

    # Create the figure and axis
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    ax[0].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')  # Base grayscale image
    ax[0].set_title('Original volume')
    ax[0].axis('off')

    # Ground truth segmentation with volume overlay
    ax[1].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')  # Base grayscale image
    ax[1].imshow(vol_gt[:, :, max_slice], cmap=cmap, interpolation='none', vmin=0, vmax=3,
                 alpha=0.5)  # Segmentation overlay with transparency
    ax[1].set_title('Ground truth segmentation')
    ax[1].axis('off')

    # Predicted segmentation with volume overlay
    ax[2].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')  # Base grayscale image
    ax[2].imshow(vol_pred[:, :, max_slice], cmap=cmap, interpolation='none', vmin=0, vmax=3,
                 alpha=0.5)  # Segmentation overlay with transparency
    ax[2].set_title('Predicted segmentation')
    ax[2].axis('off')

    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)


def save_animation(vol_og, vol_gt, vol_pred, save_path):
    '''
    Save the 3D volume as an animation (GIF).
    ------------------------------------------------
    Parameters:
    -----------------------------------------------
    - vol_og: 3D numpy array: Channel from the input tensor
    - vol_gt: 3D numpy array: Ground truth segmentation volume
    - vol_pred: 3D numpy array: Predicted segmentation volume
    - save_path: str: Path to save the animation
    -----------------------------------------------
    Returns:
    - None
    '''

    # Define the colors for each label (0: background, 1: non-enhancing, 2: whole tumor, 3: enhancing tumor)
    colors = [(0, 0, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255)]  # RGB values: black, yellow, red, cyan
    cmap = mcolors.ListedColormap(np.array(colors) / 255.0)  # Normalize to [0,1] for matplotlib

    # Get the number of slices (assumes all volumes have the same shape)
    _, _, num_slices = vol_og.shape

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    # Function to update the figure for each frame (slice)
    def update(i):
        for ax in axes:
            ax.clear()  # Clear the previous frame

        # Display the original volume (grayscale)
        axes[0].imshow(vol_og[:, :, i], cmap='gray')
        axes[0].set_title('Original Volume')

        # Display the ground truth segmentation with the custom colormap and alpha 0.5
        axes[1].imshow(vol_og[:, :, i], cmap='gray')
        axes[1].imshow(vol_gt[:, :, i], cmap=cmap, alpha=0.5)
        axes[1].set_title('Ground Truth')

        # Display the predicted segmentation with the custom colormap and alpha 0.5
        axes[2].imshow(vol_og[:, :, i], cmap='gray')
        axes[2].imshow(vol_pred[:, :, i], cmap=cmap, alpha=0.5)
        axes[2].set_title('Predicted')

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_slices, repeat=True)

    # Save the animation as a GIF
    ani.save(save_path, writer='pillow', fps=10)

    plt.close(fig)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Inference script for SegFormer3D on BraTS 2017 dataset")
    parser.add_argument(
        "--inf_all",
        action="store_true",
        help="If set, perform inference on all validation cases.",
    )
    parser.add_argument(
        "--case_number",
        type=int,
        default=1,
        help="Case number of the patient to perform inference on (1-484).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights file (.pth). If not provided, uses default weights.",
    )
    args = parser.parse_args()
    case_number = args.case_number
    inf_all = args.inf_all
    # Device setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    this_file_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(this_file_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    grand_grandparent_dir = os.path.dirname(grandparent_dir)

    config_path = os.path.join(
        parent_dir,
        "experiments/brats_2017/template_experiment/config.yaml"
    )
    
    # Use provided weights path or default
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = os.path.join(
            parent_dir,
            "best_segformer3d_brats_performance.pth"
        )
    # weights_path = os.path.join(this_file_dir, 'model_checkpoints', 'best_dice_checkpoint', 'pytorch_model.bin')

    # Model configuration
    model_config = load_config(config_path)

    # Build the SegFormer architecture
    model = build_architecture(model_config)
    model.to(device)

    # Declare model weights
    initial_weights = model.state_dict()

    # Load weights
    model_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(model_dict, strict=False)

    print("Model loaded with weights from:", weights_path)

    # Inference
    model.eval()

    if inf_all:
        # Paths to the tensor and label files
        data_path = os.path.join(parent_dir, 'data')
        brats2017_seg_path = os.path.join(data_path, 'brats2017_seg')
        val_csv_path = os.path.join(brats2017_seg_path, 'validation.csv')
        val_df = pd.read_csv(val_csv_path)

        val_paths = val_df['data_path'].to_numpy()
        val_cases = val_df['case_name'].to_numpy()

        plots_folder = os.path.join("/app/data/output", 'inference_plots')
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        with tqdm(total=len(val_paths)) as pbar:
            for i in range(len(val_paths)):

                # Extract the tensor and label paths for the i-th case in the validation set
                val_path_i = val_paths[i]
                val_case_i = val_cases[i]

                # Load preprocessed data from .pt files (same as evaluate.py)
                case_path = os.path.join(brats2017_seg_path, f"BraTS2017_Training_Data/{val_case_i}")
                volume_fp = os.path.join(case_path, f"{val_case_i}_modalities.pt")
                label_fp = os.path.join(case_path, f"{val_case_i}_label.pt")

                try:
                    # Load preprocessed volume and label
                    volume = torch.load(volume_fp, map_location=device, weights_only=False)
                    label = torch.load(label_fp, map_location=device, weights_only=False)

                    # Convert to tensors if needed
                    if not isinstance(volume, torch.Tensor):
                        volume = torch.from_numpy(volume)
                    if not isinstance(label, torch.Tensor):
                        label = torch.from_numpy(label)

                    # For inference, use preprocessed data directly
                    input_tensor = volume.float()
                    label_tensor = label.float()
                    
                    # For visualization, inverse the swapaxes(1,3) from preprocessing
                    input_np = volume.cpu().numpy()
                    input_np = np.swapaxes(input_np, 1, 3)  # Restore original orientation
                    label_np = label.cpu().numpy()
                    label_np = np.swapaxes(label_np, 1, 3)  # Restore original orientation

                except Exception as e:
                    print(f"Error loading data for {val_case_i}: {str(e)}")
                    pbar.update(1)
                    continue

                # Add a batch dimension to the input and label tensors for the model
                input_tensor = input_tensor.unsqueeze(0).to(device)
                label_tensor = label_tensor.unsqueeze(0).to(device)

                # ---------------------------- #

                # Define the post-transforms for the predictions
                post_transform = Compose([
                    Activations(sigmoid=True),  # Sigmoid activation to the model output
                    AsDiscrete(argmax=False, threshold=0.5)  # Thresholding the output]
                ]
                )

                # Get the predicted segmentation using sliding window inference from MONAI
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=[128, 128, 128],
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.5,
                )

                decollated_preds = decollate_batch(logits)  # Decollate the batch of predictions

                # Convert the predictions to the final output format
                output_convert = [
                    post_transform(val_pred_tensor) for val_pred_tensor in decollated_preds
                ]

                # Get the final prediction volume
                final_pred = output_convert[0]
                final_pred_np = final_pred.cpu().numpy()  # Convert the tensor to numpy array
                
                # Inverse swapaxes on predictions to match visualization space
                final_pred_np = np.swapaxes(final_pred_np, 1, 3)
                
                final_pred_np = final_pred_np > 0.5  # Threshold the output to get the final segmentation (0.5 threshold because of the sigmoid activation)

                # #Label 0: Background
                # #Label 1: Non-enhancing tumor (NCR/NET)
                # #Label 2: WT (whole tumor) = ET (enhancing tumor) + NCR/NET (non-enhancing tumor)
                # #Label 3: ET (enhancing tumor)

                class1_vol, class2_vol, class3_vol = final_pred_np  # Split the final prediction volume into the 3 classes

                # print(np.sum(class1_vol))
                # print(np.sum(class2_vol))
                # print(np.sum(class3_vol))

                # Combine the 3 classes to get the final prediction volume (0: background, 1: non-enhancing, 2: whole tumor, 3: enhancing tumor)
                final_pred_vol = np.zeros_like(class1_vol)
                final_pred_vol = np.sum([class1_vol, class2_vol, class3_vol], axis=0)

                # ------- Fused the predictions of 4 channels to obtain a 3D segmentation volume --------
                final_pred_vol = final_pred_vol.astype(np.uint8)
                # -----------------------------#

                # Label volume creation from .pt format [TC, WT, ET]
                gt_final_vol = np.zeros_like(final_pred_vol)

                tc_np, wt_np, et_np = label_np  # Split: TC, WT, ET

                # Create visualization labels prioritizing visibility:
                # Label 1: WT (Whole Tumor) - yellow
                # Label 2: TC (Tumor Core) - red  
                # Label 3: ET (Enhancing Tumor) - cyan
                gt_final_vol[wt_np > 0.5] = 1  # Whole tumor
                gt_final_vol[tc_np > 0.5] = 2  # Tumor core
                gt_final_vol[et_np > 0.5] = 3  # Enhancing tumor

                # -----------------------------#

                # Extract a channel from the input tensor for visualization
                input_np_channel = input_np[0]  # Directly use the data from the first channel for visualization.

                # -----------------------------#

                case_plots_folder = os.path.join(plots_folder, val_case_i)
                if not os.path.exists(case_plots_folder):
                    os.makedirs(case_plots_folder)

                # -----------------------------#

                max_slice, max_area = get_max_slice(final_pred_vol)

                save_img_path = os.path.join(case_plots_folder, val_case_i + '_max_area.png')
                save_animation_path = os.path.join(case_plots_folder, val_case_i + '_animation.gif')
                save_np_path = os.path.join(case_plots_folder, val_case_i + '_final_pred.npy')

                save_biggest_area_img(input_np_channel, gt_final_vol, final_pred_vol, max_slice, save_img_path)
                save_animation(input_np_channel, gt_final_vol, final_pred_vol, save_animation_path)
                np.save(save_np_path, final_pred_vol)

                # print("Save animation path:", save_animation_path)
                # print("Save image path:", save_img_path)
                # print("Save numpy path:", save_np_path)

                pbar.set_description(f"Case: {val_case_i} with Max area: {max_area}")
                # print("****************")
                # time.sleep(60)
                pbar.update(1)

    else:
        plots_folder = os.path.join("/app/data/output", 'inference_plots')
        # Extract the tensor and label paths for the i-th case in the validation set
        data_path = os.path.join(parent_dir, 'data', 'brats2017_seg')
        # case number should be between 1 and 484 and translated to BRATS_xxx format
        case_name = f"BRATS_{case_number:03d}"
        print(f"Processing case: {case_name}\n")

        # Load preprocessed data from .pt files (same as evaluate.py)
        case_path = os.path.join(data_path, f"BraTS2017_Training_Data/{case_name}")
        volume_fp = os.path.join(case_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(case_path, f"{case_name}_label.pt")

        try:
            # Load preprocessed volume and label
            volume = torch.load(volume_fp, map_location=device, weights_only=False)
            label = torch.load(label_fp, map_location=device, weights_only=False)

            # Convert to tensors if needed
            if not isinstance(volume, torch.Tensor):
                volume = torch.from_numpy(volume)
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label)

            # For inference, use preprocessed data directly
            input_tensor = volume.float()
            label_tensor = label.float()
            
            # For visualization, inverse the swapaxes(1,3) from preprocessing
            input_np = volume.cpu().numpy()
            input_np = np.swapaxes(input_np, 1, 3)  # Restore original orientation
            label_np = label.cpu().numpy()
            label_np = np.swapaxes(label_np, 1, 3)  # Restore original orientation

        except Exception as e:
            print(f"Error loading data for {case_name}: {str(e)}")
            raise

        # Add a batch dimension to the input and label tensors for the model
        input_tensor = input_tensor.unsqueeze(0).to(device)
        label_tensor = label_tensor.unsqueeze(0).to(device)

        # ---------------------------- #

        # Define the post-transforms for the predictions
        post_transform = Compose([
            Activations(sigmoid=True),  # Sigmoid activation to the model output
            AsDiscrete(argmax=False, threshold=0.5)  # Thresholding the output]
        ]
        )

        # Get the predicted segmentation using sliding window inference from MONAI
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=[128, 128, 128],
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
        )

        decollated_preds = decollate_batch(logits)  # Decollate the batch of predictions

        # Convert the predictions to the final output format
        output_convert = [
            post_transform(val_pred_tensor) for val_pred_tensor in decollated_preds
        ]

        # Get the final prediction volume
        final_pred = output_convert[0]
        final_pred_np = final_pred.cpu().numpy()  # Convert the tensor to numpy array
        
        # Inverse swapaxes on predictions to match visualization space
        final_pred_np = np.swapaxes(final_pred_np, 1, 3)
        
        final_pred_np = final_pred_np > 0.5  # Threshold the output to get the final segmentation (0.5 threshold because of the sigmoid activation)

        # #Label 0: Background
        # #Label 1: Non-enhancing tumor (NCR/NET)
        # #Label 2: WT (whole tumor) = ET (enhancing tumor) + NCR/NET (non-enhancing tumor)
        # #Label 3: ET (enhancing tumor)

        class1_vol, class2_vol, class3_vol = final_pred_np  # Split the final prediction volume into the 3 classes

        # print(np.sum(class1_vol))
        # print(np.sum(class2_vol))
        # print(np.sum(class3_vol))

        # Combine the 3 classes to get the final prediction volume (0: background, 1: non-enhancing, 2: whole tumor, 3: enhancing tumor)
        final_pred_vol = np.zeros_like(class1_vol)
        final_pred_vol = np.sum([class1_vol, class2_vol, class3_vol], axis=0)

        # ------- Fused the predictions of 4 channels to obtain a 3D segmentation volume --------
        final_pred_vol = final_pred_vol.astype(np.uint8)
        # -----------------------------#

        # Label volume creation from .pt format [TC, WT, ET]
        gt_final_vol = np.zeros_like(final_pred_vol)

        tc_np, wt_np, et_np = label_np  # Split: TC, WT, ET

        # Create visualization labels prioritizing visibility:
        # Label 1: WT (Whole Tumor) - yellow
        # Label 2: TC (Tumor Core) - red  
        # Label 3: ET (Enhancing Tumor) - cyan
        gt_final_vol[wt_np > 0.5] = 1  # Whole tumor
        gt_final_vol[tc_np > 0.5] = 2  # Tumor core
        gt_final_vol[et_np > 0.5] = 3  # Enhancing tumor

        # -----------------------------#

        # Extract a channel from the input tensor for visualization
        input_np_channel = input_np[0]  # Directly use the data from the first channel for visualization.

        # -----------------------------#

        case_plots_folder = os.path.join(plots_folder, case_name)
        print("Case plots folder:", case_plots_folder)
        if not os.path.exists(case_plots_folder):
            os.makedirs(case_plots_folder)

        # -----------------------------#

        max_slice, max_area = get_max_slice(final_pred_vol)

        save_img_path = os.path.join(case_plots_folder, case_name + '_max_area.png')
        save_animation_path = os.path.join(case_plots_folder, case_name + '_animation.gif')
        save_np_path = os.path.join(case_plots_folder, case_name + '_final_pred.npy')

        print("Save animation path:", save_animation_path)
        print("Save image path:", save_img_path)
        print("Save numpy path:", save_np_path)

        save_biggest_area_img(input_np_channel, gt_final_vol, final_pred_vol, max_slice, save_img_path)
        save_animation(input_np_channel, gt_final_vol, final_pred_vol, save_animation_path)
        np.save(save_np_path, final_pred_vol)

