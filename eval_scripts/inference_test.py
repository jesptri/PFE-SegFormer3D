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
from monai.metrics import DiceMetric
import nibabel as nib

sys.path.append("../../../")

from architectures.build_architecture import build_architecture

def load_config(config_path: str) -> Dict:
    """Loads the yaml config file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_max_slice(vol):
    """Get the slice with the largest tumor area in a 3D volume"""
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
    """Save comparison image with the largest tumor area slice"""
    colors = [(0, 0, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255)]
    cmap = mcolors.ListedColormap(np.array(colors) / 255.0)

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    ax[0].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')
    ax[0].set_title('Original volume')
    ax[0].axis('off')

    ax[1].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')
    ax[1].imshow(vol_gt[:, :, max_slice], cmap=cmap, interpolation='none', 
                 vmin=0, vmax=3, alpha=0.5)
    ax[1].set_title('Ground truth segmentation')
    ax[1].axis('off')

    ax[2].imshow(vol_og[:, :, max_slice], cmap='gray', interpolation='none')
    ax[2].imshow(vol_pred[:, :, max_slice], cmap=cmap, interpolation='none', 
                 vmin=0, vmax=3, alpha=0.5)
    ax[2].set_title('Predicted segmentation')
    ax[2].axis('off')

    plt.savefig(save_path)
    plt.close(fig)


def save_animation(vol_og, vol_gt, vol_pred, save_path):
    """Save the 3D volume as an animation (GIF)"""
    colors = [(0, 0, 0), (255, 255, 0), (255, 0, 0), (0, 255, 255)]
    cmap = mcolors.ListedColormap(np.array(colors) / 255.0)

    _, _, num_slices = vol_og.shape
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    def update(i):
        for ax in axes:
            ax.clear()

        axes[0].imshow(vol_og[:, :, i], cmap='gray')
        axes[0].set_title('Original Volume')

        axes[1].imshow(vol_og[:, :, i], cmap='gray')
        axes[1].imshow(vol_gt[:, :, i], cmap=cmap, alpha=0.5)
        axes[1].set_title('Ground Truth')

        axes[2].imshow(vol_og[:, :, i], cmap='gray')
        axes[2].imshow(vol_pred[:, :, i], cmap=cmap, alpha=0.5)
        axes[2].set_title('Predicted')

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

    ani = animation.FuncAnimation(fig, update, frames=num_slices, repeat=True)
    ani.save(save_path, writer='pillow', fps=10)
    plt.close(fig)


class InferenceWithMetrics:
    """Handles inference and metric computation for validation data"""
    
    def __init__(self, roi_size, sw_batch_size):
        """Initialize inference module with metric computation
        
        Args:
            roi_size: Region of interest size for sliding window
            sw_batch_size: Batch size for sliding window patches
        """
        self.dice_metric = DiceMetric(
            include_background=False,  # Exclude background channel
            reduction="mean_batch",
            get_not_nans=False
        )
        self.post_transform = Compose([
            Activations(sigmoid=True),
            AsDiscrete(argmax=False, threshold=0.5),
        ])
        self.sw_batch_size = sw_batch_size
        self.roi_size = roi_size
    
    def compute_dice(self, val_inputs, val_labels, model):
        """Compute Dice metric for a single case
        
        Args:
            val_inputs: Input volume (1, 4, 128, 128, 128)
            val_labels: Ground truth labels (1, 3, 128, 128, 128)
            model: Segmentation model
            
        Returns:
            logits: Raw model predictions
            dice_scores: Dictionary with individual and average Dice scores
        """
        self.dice_metric.reset()
        
        # Perform sliding window inference
        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=val_inputs,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=model,
                overlap=0.5,
            )
        
        # Decollate and post-process
        val_labels_list = decollate_batch(val_labels)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [
            self.post_transform(val_pred_tensor) 
            for val_pred_tensor in val_outputs_list
        ]
        
        # Compute Dice metric
        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        
        # Get per-channel Dice scores
        dice_per_channel = self.dice_metric.aggregate().cpu().numpy()
        
        # Structure the results
        dice_scores = {
            'TC': float(dice_per_channel[0]) * 100.0,      # Tumor Core
            'WT': float(dice_per_channel[1]) * 100.0,      # Whole Tumor
            'ET': float(dice_per_channel[2]) * 100.0,      # Enhancing Tumor
            'average': float(dice_per_channel.mean()) * 100.0
        }
        
        return logits, dice_scores


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths setup
    this_file_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(this_file_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    grand_grandparent_dir = os.path.dirname(grandparent_dir)

    config_path = os.path.join(this_file_dir, "config.yaml")
    weights_path = os.path.join(grand_grandparent_dir, "best_segformer3d_brats_performance.pth")

    # Load configuration and build model
    model_config = load_config(config_path)
    model = build_architecture(model_config)
    model.to(device)

    # Load weights
    model_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    print("Model loaded with weights from:", weights_path)

    # Initialize inference module with metrics
    inference_module = InferenceWithMetrics(
        roi_size=[128, 128, 128],
        sw_batch_size=4
    )

    # Load validation data
    data_path = os.path.join(grand_grandparent_dir, 'data')
    brats2017_seg_path = os.path.join(data_path, 'brats2017_seg')
    val_csv_path = os.path.join(brats2017_seg_path, 'validation.csv')
    val_df = pd.read_csv(val_csv_path)

    val_paths = val_df['data_path'].to_numpy()
    val_cases = val_df['case_name'].to_numpy()

    # Create output directories
    plots_folder = os.path.join(this_file_dir, 'plots')
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Store results for summary statistics
    all_dice_scores = []

    print("\n" + "="*80)
    print("Starting inference with evaluation...")
    print("="*80 + "\n")

    with tqdm(total=len(val_paths), desc="Processing cases") as pbar:
        for i in range(len(val_paths)):
            val_case_i = val_cases[i]

            # Load all 4 MRI modalities
            modal_names = ['0000', '0001', '0002', '0003']
            input_np = []
            for modal in modal_names:
                modal_path = os.path.join(
                    model_config["dataset_parameters"]["brats_raw_root"], 
                    "train", "imagesTr", 
                    f"{val_case_i}_{modal}.nii.gz"
                )
                modal_full = nib.load(modal_path).get_fdata()
                modal_crop = modal_full[56:184, 56:184, 13:141]
                modal_crop = (modal_crop - modal_crop.min()) / (modal_crop.max() - modal_crop.min())
                input_np.append(modal_crop.astype(np.float32))
            input_np = np.stack(input_np)

            # Load ground truth label
            label_path = os.path.join(
                model_config["dataset_parameters"]["brats_raw_root"], 
                'train', 'labelsTr', 
                val_case_i + '.nii.gz'
            )
            label_img = nib.load(label_path)
            label_data = label_img.get_fdata()[56:184, 56:184, 13:141]

            # Convert to multi-channel based on BraTS classes (MUST MATCH TRAINING)
            # Channel 0: TC (Tumor Core) = labels 2 OR 3
            # Channel 1: WT (Whole Tumor) = labels 1 OR 2 OR 3
            # Channel 2: ET (Enhancing Tumor) = label 3 only
            tc = ((label_data == 2) | (label_data == 3)).astype(np.float32)
            wt = ((label_data == 1) | (label_data == 2) | (label_data == 3)).astype(np.float32)
            et = (label_data == 3).astype(np.float32)
            label_np = np.stack([tc, wt, et])

            # Convert to tensors
            input_tensor = torch.from_numpy(input_np).to(device).unsqueeze(0)
            label_tensor = torch.from_numpy(label_np).to(device).unsqueeze(0)

            # Debug: Check tensor shapes
            print(f"Input shape: {input_tensor.shape}")
            print(f"Label shape: {label_tensor.shape}")
            print(f"Label channels - TC: {tc.sum()}, WT: {wt.sum()}, ET: {et.sum()}")

            # ============================================================
            # INFERENCE WITH METRICS COMPUTATION
            # ============================================================
            logits, dice_scores = inference_module.compute_dice(
                input_tensor, 
                label_tensor, 
                model
            )
            
            # Store scores for summary
            all_dice_scores.append({
                'case_name': val_case_i,
                **dice_scores
            })

            # Post-process predictions for visualization
            decollated_preds = decollate_batch(logits)
            output_convert = [
                inference_module.post_transform(val_pred_tensor) 
                for val_pred_tensor in decollated_preds
            ]
            
            final_pred = output_convert[0]
            final_pred_np = final_pred.cpu().numpy()

            # Create final prediction volume
            class1_vol, class2_vol, class3_vol = final_pred_np
            final_pred_vol = np.sum([class1_vol, class2_vol, class3_vol], axis=0)
            final_pred_vol = final_pred_vol.astype(np.uint8)

            # Create ground truth volume for visualization
            # Priority: ET (3) > WT (2) > TC (1)
            gt_final_vol = np.zeros_like(final_pred_vol)
            tc_np, wt_np, et_np = label_np
            gt_final_vol[tc_np > 0.5] = 1   # Tumor Core
            gt_final_vol[wt_np > 0.5] = 2   # Whole Tumor (overwrites TC where both exist)
            gt_final_vol[et_np > 0.5] = 3   # Enhancing Tumor (highest priority)

            # Use first channel for visualization
            input_np_channel = input_np[0]

            # Create case folder
            case_plots_folder = os.path.join(plots_folder, val_case_i)
            if not os.path.exists(case_plots_folder):
                os.makedirs(case_plots_folder)

            # Find max slice and save visualizations
            max_slice, max_area = get_max_slice(final_pred_vol)

            save_img_path = os.path.join(case_plots_folder, val_case_i + '_max_area.png')
            save_animation_path = os.path.join(case_plots_folder, val_case_i + '_animation.gif')
            save_np_path = os.path.join(case_plots_folder, val_case_i + '_final_pred.npy')

            save_biggest_area_img(input_np_channel, gt_final_vol, final_pred_vol, 
                                 max_slice, save_img_path)
            save_animation(input_np_channel, gt_final_vol, final_pred_vol, 
                          save_animation_path)
            np.save(save_np_path, final_pred_vol)

            # Update progress bar with metrics
            pbar.set_postfix({
                'Dice': f"{dice_scores['average']:.2f}%",
                'Area': max_area
            })
            pbar.update(1)

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    print("\n" + "="*80)
    print("INFERENCE COMPLETE - SUMMARY STATISTICS")
    print("="*80 + "\n")

    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(all_dice_scores)
    
    # Compute overall statistics
    print("Overall Dice Scores (%):")
    print("-" * 40)
    print(f"TC (Tumor Core):            {results_df['TC'].mean():.2f} ± {results_df['TC'].std():.2f}")
    print(f"WT (Whole Tumor):           {results_df['WT'].mean():.2f} ± {results_df['WT'].std():.2f}")
    print(f"ET (Enhancing Tumor):       {results_df['ET'].mean():.2f} ± {results_df['ET'].std():.2f}")
    print(f"Average across all classes: {results_df['average'].mean():.2f} ± {results_df['average'].std():.2f}")
    print()

    # Save detailed results to CSV
    results_csv_path = os.path.join(plots_folder, 'validation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Detailed results saved to: {results_csv_path}")

    # Find best and worst cases
    best_case = results_df.loc[results_df['average'].idxmax()]
    worst_case = results_df.loc[results_df['average'].idxmin()]
    
    print("\n" + "-" * 40)
    print(f"Best case:  {best_case['case_name']} (Dice: {best_case['average']:.2f}%)")
    print(f"Worst case: {worst_case['case_name']} (Dice: {worst_case['average']:.2f}%)")
    print("=" * 80)