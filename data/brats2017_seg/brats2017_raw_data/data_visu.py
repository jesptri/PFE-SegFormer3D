import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors




def animate(input_1, input_2):
    """animate pairs of image sequences of the same length on two conjugate axis"""
    # assert len(input_1) == len(
    #     input_2
    # ), f"two inputs should have the same number of frame but first input had {len(input_1)} and the second one {len(input_2)}"
    # # set the figure and axis
    volumes = input_1[0, ...], input_1[1, ...], input_1[2, ...], input_1[3, ...]
    y1, y2, y3 = input_2[0, ...], input_2[1, ...], input_2[2, ...]
    fig, axis = plt.subplots(2, 4, figsize=(16, 8))
    title_list = ["Flair", "T1w", "T1gd", "T2w"]
    for i in range(4):
        axis[0, i].set_axis_off()
        axis[1, i].set_axis_off()
        axis[0, i].set_title(f"{title_list[i]}", fontsize=16)
        axis[1, i].set_title("Segmentation Overlay", fontsize=16)
    sequence = []
    colors_labels = {
        0: (1, 1, 0),      # Yellow for y1
        1: (1, 0, 0),      # Red for y2
        2: (0, 1, 1)       # Cyan for y3
    }

    sequence_length = max(v.shape[0] for v in volumes)
    padded_volumes = []
    for v in volumes:
        if v.shape[0] < sequence_length:
            pad_width = sequence_length - v.shape[0]
            padded_v = np.pad(v, ((0, pad_width), (0, 0), (0, 0)), mode='edge')
            padded_volumes.append(padded_v)
        else:
            padded_volumes.append(v)
    for i in range(sequence_length):
        intermediate_frames = []
        for j, v in enumerate(padded_volumes):
        
            im_1 = axis[0, j].imshow(v[i], cmap="bone", animated=True)
            im_2 = axis[1, j].imshow(v[i], cmap="bone", animated=True)
            
            # Create overlay with different colors for each label
            overlay = np.zeros((*y1[i].shape, 3), dtype=float)
            overlay[y2[i] > 0] = colors_labels[0]  # Red where y2 > 0
            # Yellow where y1 > 0
            overlay[y1[i] > 0] = colors_labels[1]
            overlay[y3[i] > 0] = colors_labels[2]  # Cyan where y3 > 0
            
            
            
            im_3 = axis[1, j].imshow(overlay, alpha=0.4, animated=True)
            intermediate_frames.extend([im_1, im_2, im_3])

        sequence.append(intermediate_frames)
    return animation.ArtistAnimation(
        fig,
        sequence,
        interval=100,
        blit=True,
        repeat_delay=100,
    )

def viz(volume, label, case_name: str = None)->None:
    """
    pair visualization of the volume and label
    volume_indx: index for the volume. ["Flair", "t1", "t1ce", "t2"]
    label_indx: index for the label segmentation ["TC" (Tumor core), "WT" (Whole tumor), "ET" (Enhancing tumor)]
    """
    x = volume
    y = label
    ani = animate(input_1=x, input_2=y)
    os.makedirs("data/visualization", exist_ok=True)
    ani.save(f"data/visualization/{case_name}_volume.gif", writer="pillow")
    plt.tight_layout()
    plt.close()
    

def animate_3views(input_1, input_2):
    """animate 3 orthogonal views (axial, coronal, sagittal) of image sequences"""
    y1, y2, y3 = input_2[0, ...], input_2[1, ...], input_2[2, ...]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flatten():
        ax.set_axis_off()
    
    # Titles
    axes[0, 0].set_title("Axial - MRI", fontsize=14)
    axes[0, 1].set_title("Coronal - MRI", fontsize=14)
    axes[0, 2].set_title("Sagittal - MRI", fontsize=14)
    axes[1, 0].set_title("Axial - Overlay", fontsize=14)
    axes[1, 1].set_title("Coronal - Overlay", fontsize=14)
    axes[1, 2].set_title("Sagittal - Overlay", fontsize=14)
    
    sequence_length = input_1.shape[0]
    sequence = []
    colors_labels = {
        0: (1, 1, 0),      # Yellow for y1 (TC)
        1: (1, 0, 0),      # Red for y2 (WT)
        2: (0, 1, 1)       # Cyan for y3 (ET)
    }
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, weight='bold')
    
    for i in range(sequence_length):
        # Axial view (input_1[i, :, :])
        im_1_axial = axes[0, 0].imshow(input_1[i, :, :], cmap="bone", animated=True)
        
        # Coronal view (input_1[:, i, :])
        im_1_coronal = axes[0, 1].imshow(input_1[:, i, :], cmap="bone", animated=True)
        
        # Sagittal view (input_1[:, :, i])
        im_1_sagittal = axes[0, 2].imshow(input_1[:, :, i], cmap="bone", animated=True)
        
        # Create overlays for each view
        # Axial
        overlay_axial = np.zeros((*y1[i, :, :].shape, 3), dtype=float)
        overlay_axial[y2[i, :, :] > 0] = colors_labels[1]  # Red where y2 > 0
        overlay_axial[y1[i, :, :] > 0] = colors_labels[0]  # Yellow where y1 > 0
        overlay_axial[y3[i, :, :] > 0] = colors_labels[2]  # Cyan where y3 > 0
        im_2_axial = axes[1, 0].imshow(input_1[i, :, :], cmap="bone", animated=True)
        im_3_axial = axes[1, 0].imshow(overlay_axial, alpha=0.5, animated=True)
        
        # Coronal
        overlay_coronal = np.zeros((*y1[:, i, :].shape, 3), dtype=float)
        overlay_coronal[y2[:, i, :] > 0] = colors_labels[1]
        overlay_coronal[y1[:, i, :] > 0] = colors_labels[0]
        overlay_coronal[y3[:, i, :] > 0] = colors_labels[2]
        im_2_coronal = axes[1, 1].imshow(input_1[:, i, :], cmap="bone", animated=True)
        im_3_coronal = axes[1, 1].imshow(overlay_coronal, alpha=0.5, animated=True)
        
        # Sagittal
        overlay_sagittal = np.zeros((*y1[:, :, i].shape, 3), dtype=float)
        overlay_sagittal[y2[:, :, i] > 0] = colors_labels[1]
        overlay_sagittal[y1[:, :, i] > 0] = colors_labels[0]
        overlay_sagittal[y3[:, :, i] > 0] = colors_labels[2]
        im_2_sagittal = axes[1, 2].imshow(input_1[:, :, i], cmap="bone", animated=True)
        im_3_sagittal = axes[1, 2].imshow(overlay_sagittal, alpha=0.5, animated=True)
        
        frame_text.set_text(f'Frame: {i + 1} / {sequence_length}')

        sequence.append([im_1_axial, im_1_coronal, im_1_sagittal, 
                        im_2_axial, im_3_axial, im_2_coronal, im_3_coronal, 
                        im_2_sagittal, im_3_sagittal, frame_text])
    
    return animation.ArtistAnimation(
        fig,
        sequence,
        interval=100,
        blit=True,
        repeat_delay=100,
    )

def main():
    import argparse
    from brats2017_raw_dataset import Brats2017RawDataset
    parser = argparse.ArgumentParser(
        description="Visualize BraTS 2017 dataset cases."
    )
    parser.add_argument(
        "case_number", type=int, default=1,
        help="Case number to visualize (1-484)."
    )
    args = parser.parse_args()
    dataset = Brats2017RawDataset()
    patient_data = dataset[args.case_number]
    print(f"Visualizing case number: {args.case_number}")
    from data_visu import viz
    volume, label, case_name = patient_data
    viz(volume, label, case_name=case_name)

if __name__ == "__main__":
    main()