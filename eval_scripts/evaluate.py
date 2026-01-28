import os
import sys
import yaml
from tqdm import tqdm
from typing import Dict

import torch
import numpy as np
import pandas as pd
import nibabel as nib

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
from monai.metrics import DiceMetric

sys.path.append("../")

from architectures.build_architecture import build_architecture

import warnings

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class EvaluationModule:
    """Fast evaluation module without visualization overhead"""
    
    def __init__(self, roi_size, sw_batch_size):
        """Initialize evaluation module
        
        Args:
            roi_size: Region of interest size for sliding window
            sw_batch_size: Batch size for sliding window patches
        """
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean_batch",
            get_not_nans=False
        )
        self.post_transform = Compose([
            Activations(sigmoid=True),
            AsDiscrete(argmax=False, threshold=0.5),
        ])
        self.sw_batch_size = sw_batch_size
        self.roi_size = roi_size
    
    def evaluate_case(self, val_inputs, val_labels, model):
        """
        Ignore Dice for classes absent in GT (BraTS-compliant evaluation)
        """
        self.dice_metric.reset()

        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=val_inputs,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=model,
                overlap=0.5,
            )

        val_labels_list = decollate_batch(val_labels)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [
            self.post_transform(pred) for pred in val_outputs_list
        ]

        # Compute Dice per channel (no reduction)
        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        dice = self.dice_metric.aggregate()  # shape: (B=1, C=3)

        #dice = dice[0]  # (3,)

        # Detect absent classes in GT
        gt = val_labels[0]  # (3, D, H, W)
        present_mask = torch.tensor(
            [(gt[c].sum() > 0) for c in range(gt.shape[0])],
            device=dice.device
        )

        # Mask absent classes
        dice_valid = dice[present_mask]

        # Safety: if no class present (should not happen in BraTS)
        if dice_valid.numel() == 0:
            avg_dice = torch.tensor(0.0, device=dice.device)
        else:
            avg_dice = dice_valid.mean()

        # Per-class reporting (NaN for absent classes)
        dice_out = dice.clone()
        dice_out[~present_mask] = torch.nan

        return {
            'TC': float(dice_out[0] * 100.0) if present_mask[0] else np.nan,
            'WT': float(dice_out[1] * 100.0) if present_mask[1] else np.nan,
            'ET': float(dice_out[2] * 100.0) if present_mask[2] else np.nan,
            'average': float(avg_dice * 100.0)
        }

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path setup
    this_file_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(this_file_dir)
    
    config_path = os.path.join(parent_dir, "experiments/brats_2017/template_experiment/config.yaml")
    weights_path = os.path.join(parent_dir, "best_segformer3d_brats_performance.pth")
    
    # Load model
    print("Loading model...")
    model_config = load_config(config_path)
    model = build_architecture(model_config)
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()
    print(f"Model loaded from: {weights_path}\n")
    
    # Initialize evaluation module
    evaluator = EvaluationModule(
        roi_size=[128, 128, 128],
        sw_batch_size=4
    )
    
    # Load validation and training cases
    data_path = os.path.join(parent_dir, 'data', 'brats2017_seg')
    
    val_csv_path = os.path.join(data_path, 'validation.csv')
    val_df = pd.read_csv(val_csv_path)
    val_cases = val_df['case_name'].to_numpy()
    
    train_csv_path = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    train_cases = train_df['case_name'].to_numpy()
    
    # Combine all cases
    all_cases = np.concatenate([train_cases, val_cases])
    
    print(f"Evaluating {len(train_cases)} training cases + {len(val_cases)} validation cases")
    print(f"Total: {len(all_cases)} cases")
    print("=" * 80 + "\n")
    
    # Evaluate all cases
    results = []
    
    with tqdm(total=len(all_cases), desc="Evaluation") as pbar:
        for case_name in all_cases:
            # Determine if this is a training or validation case
            split = 'train' if case_name in train_cases else 'validation'
            # Load case data
            path = os.path.join(data_path, f"BraTS2017_Training_Data/{case_name}")
            volume_fp = os.path.join(path, f"{case_name}_modalities.pt")
            label_fp = os.path.join(path, f"{case_name}_label.pt")
        
            try:
                # Use weights_only=True for security and faster loading
                # map_location='cpu' avoids unnecessary GPU allocation during loading
                volume = torch.load(volume_fp, map_location=device, weights_only=False)
                label = torch.load(label_fp, map_location=device, weights_only=False)
            
                # Convert to float32 tensors efficiently
                if not isinstance(volume, torch.Tensor):
                    volume = torch.from_numpy(volume)
                if not isinstance(label, torch.Tensor):
                    label = torch.from_numpy(label)
                
                data = {
                    "image": volume.float(),
                    "label": label.float()
                }
            except Exception as e:
                warnings.warn(f"Error loading data at ({case_name}): {str(e)}")
                # Return a fallback sample or re-raise
                raise



            # Convert to tensors and add batch dimension
            input_tensor = data["image"].unsqueeze(0)
            label_tensor = data["label"].unsqueeze(0)
            
            # Evaluate
            dice_scores = evaluator.evaluate_case(input_tensor, label_tensor, model)
            
            # Store results
            results.append({
                'case_name': case_name,
                'split': split,
                **dice_scores
            })
            
            # Update progress
            pbar.set_postfix({
                'TC': f"{dice_scores['TC']:.1f}",
                'WT': f"{dice_scores['WT']:.1f}",
                'ET': f"{dice_scores['ET']:.1f}",
                'Avg': f"{dice_scores['average']:.1f}"
            })
            pbar.update(1)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Separate train and validation results
    train_results = results_df[results_df['split'] == 'train']
    val_results = results_df[results_df['split'] == 'validation']
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - SUMMARY STATISTICS")
    print("=" * 80 + "\n")
    
    # Overall statistics
    print("OVERALL (Train + Validation):")
    print("-" * 50)
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 50)
    
    for metric in ['TC', 'WT', 'ET', 'average']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        
        label = {
            'TC': 'Tumor Core',
            'WT': 'Whole Tumor',
            'ET': 'Enhancing Tumor',
            'average': 'Average'
        }[metric]
        
        print(f"{label:<20} {mean_val:>6.2f} ± {std_val:<5.2f} {min_val:>6.2f}      {max_val:>6.2f}")
    
    print()
    
    # Training set statistics
    print("TRAINING SET:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 50)
    
    for metric in ['TC', 'WT', 'ET', 'average']:
        mean_val = train_results[metric].mean()
        std_val = train_results[metric].std()
        min_val = train_results[metric].min()
        max_val = train_results[metric].max()
        
        label = {
            'TC': 'Tumor Core',
            'WT': 'Whole Tumor',
            'ET': 'Enhancing Tumor',
            'average': 'Average'
        }[metric]
        
        print(f"{label:<20} {mean_val:>6.2f} ± {std_val:<5.2f} {min_val:>6.2f}      {max_val:>6.2f}")
    
    print()
    
    # Validation set statistics
    print("VALIDATION SET:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 50)
    
    for metric in ['TC', 'WT', 'ET', 'average']:
        mean_val = val_results[metric].mean()
        std_val = val_results[metric].std()
        min_val = val_results[metric].min()
        max_val = val_results[metric].max()
        
        label = {
            'TC': 'Tumor Core',
            'WT': 'Whole Tumor',
            'ET': 'Enhancing Tumor',
            'average': 'Average'
        }[metric]
        
        print(f"{label:<20} {mean_val:>6.2f} ± {std_val:<5.2f} {min_val:>6.2f}      {max_val:>6.2f}")
    
    print("-" * 50)
    
    # Best and worst cases
    best_idx = results_df['average'].idxmax()
    worst_idx = results_df['average'].idxmin()
    
    print(f"\nBest case (Overall):  {results_df.loc[best_idx, 'case_name']} [{results_df.loc[best_idx, 'split']}]")
    print(f"  TC: {results_df.loc[best_idx, 'TC']:.2f}%, "
          f"WT: {results_df.loc[best_idx, 'WT']:.2f}%, "
          f"ET: {results_df.loc[best_idx, 'ET']:.2f}%, "
          f"Avg: {results_df.loc[best_idx, 'average']:.2f}%")
    
    print(f"\nWorst case (Overall): {results_df.loc[worst_idx, 'case_name']} [{results_df.loc[worst_idx, 'split']}]")
    print(f"  TC: {results_df.loc[worst_idx, 'TC']:.2f}%, "
          f"WT: {results_df.loc[worst_idx, 'WT']:.2f}%, "
          f"ET: {results_df.loc[worst_idx, 'ET']:.2f}%, "
          f"Avg: {results_df.loc[worst_idx, 'average']:.2f}%")
    
    # Best and worst for validation set only
    if len(val_results) > 0:
        best_val_idx = val_results['average'].idxmax()
        worst_val_idx = val_results['average'].idxmin()
        
        print(f"\nBest case (Validation only):  {val_results.loc[best_val_idx, 'case_name']}")
        print(f"  TC: {val_results.loc[best_val_idx, 'TC']:.2f}%, "
              f"WT: {val_results.loc[best_val_idx, 'WT']:.2f}%, "
              f"ET: {val_results.loc[best_val_idx, 'ET']:.2f}%, "
              f"Avg: {val_results.loc[best_val_idx, 'average']:.2f}%")
        
        print(f"\nWorst case (Validation only): {val_results.loc[worst_val_idx, 'case_name']}")
        print(f"  TC: {val_results.loc[worst_val_idx, 'TC']:.2f}%, "
              f"WT: {val_results.loc[worst_val_idx, 'WT']:.2f}%, "
              f"ET: {val_results.loc[worst_val_idx, 'ET']:.2f}%, "
              f"Avg: {val_results.loc[worst_val_idx, 'average']:.2f}%")
    
    # Save results
    output_dir = os.path.join(this_file_dir, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results
    csv_path = os.path.join(output_dir, 'all_scores.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Save separate files for train and validation
    train_csv_path = os.path.join(output_dir, 'train_scores.csv')
    val_csv_path = os.path.join(output_dir, 'validation_scores.csv')
    train_results.to_csv(train_csv_path, index=False)
    val_results.to_csv(val_csv_path, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to:")
    print(f"  - All cases:        {csv_path}")
    print(f"  - Training set:     {train_csv_path}")
    print(f"  - Validation set:   {val_csv_path}")
    print("=" * 80)
