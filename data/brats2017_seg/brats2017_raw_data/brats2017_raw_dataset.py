import os
import numpy as np
from monai.data import MetaTensor
import nibabel
from sklearn.preprocessing import MinMaxScaler 
from monai.transforms import Orientation, EnsureType

from brats2017_seg_preprocess import ConvertToMultiChannelBasedOnBrats2017Classes

class Brats2017RawDataset():
    """ This class handles the BraTS 2017 raw dataset loading and processing."""

    def __init__(
            self,
            raw_data_path: str = "data/brats2017_seg/brats2017_raw_data/train"
        ) -> None:

        self.train_data_path = os.path.join(raw_data_path, "imagesTr")
        self.label_data_path = os.path.join(raw_data_path, "labelsTr")

        self.MRI_CODE = {
            "Flair": "0000",
            "T1w": "0001",
            "T1gd": "0002",
            "T2w": "0003",
            "label": "label"}
        
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> dict:
        """ Load the dataset file paths."""
        dataset = {}

        for i in range (1, 485):
            dataset[i] = {}

        for filename in os.listdir(self.train_data_path):
            if filename.endswith(".nii.gz"):
                patient_id = filename.split("_")[1].split(".")[0]
                modality = filename.split("_")[2].split(".")[0]
                image_path = os.path.join(self.train_data_path, filename)
                label_path = os.path.join(self.label_data_path, f"BRATS_{patient_id}.nii.gz")
                dataset[int(patient_id)][modality] = image_path
                dataset[int(patient_id)]["label"] = label_path
        
        # Check dataset integrity
        for patient_id, modalities in dataset.items():
            # Replace modality codes with descriptive names
            dataset[patient_id] = {
                "Flair": dataset[patient_id].get("0000"),
                "T1w": dataset[patient_id].get("0001"),
                "T1gd": dataset[patient_id].get("0002"),
                "T2w": dataset[patient_id].get("0003"),
                "label": dataset[patient_id].get("label")
            }
            assert len(modalities) == 5, f"Patient {patient_id} does not have all modalities."
        
        return dataset
    
    def __len__(self) -> int:
        """ Return the number of patients in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, case_name: int) -> tuple:
        code = "Flair"   #self.MRI_CODE["Flair"]
        flair = self.get_modality_fp(case_name, code)
        Flair = self.preprocess_brats_modality(flair, is_label=False)
        print(Flair.shape)
        flair_transv = Flair.swapaxes(1, 3) # transverse plane

        
        # preprocess T1w modality
        code = "T1w"   #self.MRI_CODE["T1w"]
        t1w = self.get_modality_fp(case_name, code)
        t1w = self.preprocess_brats_modality(t1w, is_label=False)
        print(t1w.shape)
        t1w_transv = t1w.swapaxes(1, 3) # transverse plane
        
        # preprocess T1gd modality
        code = "T1gd"   #self.MRI_CODE["T1gd"]
        t1gd = self.get_modality_fp(case_name, code)
        t1gd = self.preprocess_brats_modality(t1gd, is_label=False)
        t1gd_transv = t1gd.swapaxes(1, 3) # transverse plane

        
        # preprocess T2w
        code = "T2w"   #self.MRI_CODE["T2w"]
        t2w = self.get_modality_fp(case_name, code)
        t2w = self.preprocess_brats_modality(t2w, is_label=False)
        t2w_transv = t2w.swapaxes(1, 3) # transverse plane


        # preprocess segmentation label
        code = "label"   #self.MRI_CODE["label"]
        label = self.get_modality_fp(case_name, code)
        label = self.preprocess_brats_modality(label, is_label=True)
        label = label.swapaxes(1, 3) # transverse plane 

        # stack modalities (4, D, H, W)
        modalities = np.concatenate(
            (flair_transv, t1w_transv, t1gd_transv, t2w_transv),
            axis=0,
            dtype=np.float32,
        )
    
        return modalities, label, case_name

    def load_nifti(self, fp):
        """
        load a nifti file
        fp: path to the nifti file with (nii or nii.gz) extension
        """
        nifti_data = nibabel.load(fp)
        # get the floating point array
        nifti_scan = nifti_data.get_fdata()
        # get affine matrix
        affine = nifti_data.affine
        return nifti_scan, affine
    
    def _2metaTensor(self, nifti_data: np.ndarray, affine_mat: np.ndarray):
        """
        convert a nifti data to meta tensor
        nifti_data: floating point array of the raw nifti object
        affine_mat: affine matrix to be appended to the meta tensor for later application such as transformation
        """
        # creating a meta tensor in which affine matrix is stored for later uses(i.e. transformation)
        scan = MetaTensor(x=nifti_data, affine=affine_mat)
        # adding a new axis
        D, H, W = scan.shape
        # adding new axis
        scan = scan.view(1, D, H, W)
        return scan
    
    def normalize(self, x:np.ndarray)->np.ndarray:
        # Transform features by scaling each feature to a given range.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # (H, W, D) -> (H * W, D)
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data

    def orient(self, x: MetaTensor) -> MetaTensor:
        # orient the array to be in (Right, Anterior, Superior) scanner coordinate systems
        assert type(x) == MetaTensor
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)
    
    def crop_brats2021_zero_pixels(self, x: np.ndarray)->np.ndarray:
        # get rid of the zero pixels around mri scan and cut it so that the region is useful
        # crop (240, 240, 155) to (128, 128, 128)
        return x[:, 56:184, 56:184, 13:141]

    def get_modality_fp(self, case_name: int, modality: str) -> str:
        """ Get the file path for a specific modality of a patient."""
        return self.dataset[case_name][modality]

    def preprocess_brats_modality(self, data_fp: str, is_label: bool = False)->np.ndarray:
        """
        apply preprocess stage to the modality
        data_fp: directory to the modality
        """
        data, affine = self.load_nifti(data_fp)
        # label do not the be normalized 
        if is_label:
            # Binary mask does not need to be float64! For saving storage purposes!
            data = data.astype(np.uint8)
            # categorical -> one-hot-encoded 
            # (240, 240, 155) -> (3, 240, 240, 155)
            data = ConvertToMultiChannelBasedOnBrats2017Classes()(data)
        else:
            data = self.normalize(x=data)
            # (240, 240, 155) -> (1, 240, 240, 155)
            data = data[np.newaxis, ...]
        
        data = MetaTensor(x=data, affine=affine)
        # for oreinting the coordinate system we need the affine matrix
        data = self.orient(data)
        # detaching the meta values from the oriented array
        data = self.detach_meta(data)
        # (240, 240, 155) -> (128, 128, 128)
        data = self.crop_brats2021_zero_pixels(data)
        return data
    
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Brats2017RawDataset Visualization")
    parser.add_argument(
        "case_number",
        type=int,
        default=1,
        help="Case number of the patient to visualize (1-484)",
    )
    args = parser.parse_args() 

    dataset = Brats2017RawDataset()
    print(f"Total patients in dataset: {len(dataset)}")

    patient_data = dataset[args.case_number]
    print("Patient data shapes:"
          f"\n Modalities shape: {patient_data[0].shape}"
          f"\n Label shape: {patient_data[1].shape}")
    

if __name__ == "__main__":
    main()