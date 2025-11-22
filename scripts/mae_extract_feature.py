import os
import numpy as np
import torch
import argparse
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEModel, VideoMAEImageProcessor
import os.path as osp
import sys

# --- PATH FIX ---------------------------------------------------------------
# This adds the parent directory to Python's path
# so it can find the 'utils' folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.helpers import sliding_window_for_list, read_video, get_img_list

# --- GLOBAL SETTINGS --------------------------------------------------------
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
# Set benchmark to False for variable input sizes (e.g., different video clip counts)
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
class VideoMAEFeatureReader(object):
    """
    This class holds the VideoMAE model ("the Chef").
    It's responsible for the fast, GPU-bound computation.
    """
    def __init__(self, model_name, device, overlap_size, nth_layer, cache_dir=None):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def get_feats(self, video_batch):
        """
        Processes a batch of raw PIL Image clips and returns feature vectors.
        """
        # 1. Transform the images (This is the "transform" step)
        inputs = self.image_processor(images=video_batch, return_tensors="pt")

        # 2. Move tensors to GPU (non_blocking works with pin_memory=True)
        # The 'inputs' object has its own .to() method
        inputs = inputs.to(self.device, non_blocking=True)

        # 3. Run the model
        # AMP enabled
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        
        # 4. Get the last hidden state's [CLS] token
        feats = outputs[self.nth_layer][:, 0]
        return feats


# ----------------------------------------------------------------------------
class VideoDataset(Dataset):
    """
    CPU dataset for VideoMAE feature extraction.
    Loads frames, resizes to 256x256, center-crops to 224x224.
    """

    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        # Load annotation file (npy dict)
        self.data = np.load(
            osp.join(args.anno_root, f"{mode}_info.npy"),
            allow_pickle=True
        ).item()

        self.num_videos = len(self.data)
        self.ds_name = osp.split(args.anno_root)[-1]

        print(f"[VideoDataset] '{mode}' initialized with {self.num_videos} videos.")

        # --------------------------------------------------------
        # Preprocessing: resize to 256 → center crop to 224
        # --------------------------------------------------------
        self.resize_size = 256
        self.crop_size = 224

    def __len__(self):
        return self.num_videos

    # ------------------------------------------------------------
    # Helper: Resize + Center Crop
    # ------------------------------------------------------------
    def process_frame(self, img):
        # Resize to 256×256 (fast bilinear)
        img = img.resize(
            (self.resize_size, self.resize_size),
            resample=Image.BILINEAR  # Safe for all Pillow versions
        )

        # Center crop to 224×224
        left = (self.resize_size - self.crop_size) // 2
        top = (self.resize_size - self.crop_size) // 2
        right = left + self.crop_size
        bottom = top + self.crop_size
        img = img.crop((left, top, right, bottom))

        return img

    # ------------------------------------------------------------
    # Main loader
    # ------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns:
            videos: list of clips (each clip = list of 16 processed PIL images)
            fileid: ID for saving the features
            start_time_str: ONLY for How2Sign (None or number)
        """
        entry = self.data[idx]
        fname, fileid = entry["folder"], entry["fileid"]
        start_time_str = None
        videos = []

        # ============================================================
        # CASE 1: Phoenix14T / CSL-Daily (image lists)
        # ============================================================
        if self.ds_name in ["Phoenix14T", "CSL-Daily"]:

            image_list = get_img_list(self.ds_name, self.args.video_root, fname)

            # Pad to at least 16 frames
            if len(image_list) < 16:
                image_list += [image_list[-1]] * (16 - len(image_list))

            clips = sliding_window_for_list(image_list, 16, self.args.overlap_size)

            for clip in clips:
                pil_frames = []
                for path in clip:
                    try:
                        img = Image.open(path).convert("RGB")
                        img = self.process_frame(img)  # <---- APPLY RESIZE+ CROP
                        pil_frames.append(img.copy())
                        img.close()
                    except Exception as e:
                        print(f"Warning: Failed to load image {path}: {e}")
                        continue

                if len(pil_frames) > 0:
                    videos.append(pil_frames)

        # ============================================================
        # CASE 2: How2Sign (video with aligned timestamps)
        # ============================================================
        elif self.ds_name == "How2Sign":

            # Get aligned timestamps
            start_val = entry["original_info"]["START_REALIGNED"]
            end_val   = entry["original_info"]["END_REALIGNED"]

            # Safe float conversion
            try:
                start_f = float(start_val)
            except (ValueError, TypeError):
                start_f = None

            try:
                end_f = float(end_val)
            except (ValueError, TypeError):
                end_f = None

            # Read video segment
            frames = read_video(fname, start_time=start_f, end_time=end_f)

            start_time_str = str(start_f) if start_f is not None else "None"

            if len(frames) == 0:
                return [], fileid, start_time_str

            # Pad if fewer than 16 frames
            if len(frames) < 16:
                frames += [frames[-1]] * (16 - len(frames))

            # Convert frames → PIL → resize+crop
            processed_frames = []
            for f in frames:
                img = Image.fromarray(f).convert("RGB")
                img = self.process_frame(img)          # <---- APPLY RESIZE+ CROP
                processed_frames.append(img)

            # Now slide over the processed frames
            videos = sliding_window_for_list(processed_frames, 16, self.args.overlap_size)

        else:
            raise NotImplementedError(f"Unknown dataset: {self.ds_name}")

        return videos, fileid, start_time_str


# ----------------------------------------------------------------------------
def custom_collate_fn(batch):
    """
    This collate function is necessary because the default collate
    doesn't know how to handle PIL Images.
    Since DataLoader batch_size=1, 'batch' is a list with one item:
    [ (list_of_pil_clips, "file_id", "start_time") ]
    We just unpack and return that single item.
    """
    return batch[0]
# ----------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', required=True)
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the *GPU* (model processing)")
    # FIX: Smart default for device
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--mode', nargs='+', type=str)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    return parser


# ----------------------------------------------------------------------------
def main():
    args = get_parser().parse_args()

    # 1. Create the "Chef" (GPU-side model)
    reader = VideoMAEFeatureReader(
        args.model_name,
        args.device,
        args.overlap_size,
        args.nth_layer,
        args.cache_dir
    )

    mode = ["test"]
    for m in mode:
        ds_name = osp.split(args.anno_root)[-1]
        out_folder = f"mae_feat_{ds_name}"
        
        # FIX: Use _m for the save path
        if ds_name == "How2Sign":    _m = "val" if m == "dev" else m
        elif ds_name == "NIASL2021": _m = "validation" if m == "dev" else m
        else:                       _m = m
        
        # Create the save directory using the correct split name
        save_dir_split = osp.join(args.save_dir, out_folder, _m)
        os.makedirs(save_dir_split, exist_ok=True)

        # 2. Create the "Waiter" (CPU-side dataset)
        dataset = VideoDataset(args, _m)

        # 3. Create the "Waiter Team Manager" (DataLoader)
        dataloader = DataLoader(
            dataset,
            batch_size=1,           # must be 1
            shuffle=False,
            collate_fn=custom_collate_fn, # Use our custom collate
            num_workers=args.num_workers, # Parallel CPU workers
            pin_memory=True,              # Fast CPU-to-GPU transfer
            persistent_workers=True   # Keep workers alive
        )
     
        print(f"Extracting '{_m}' using {args.num_workers} workers... Saving to {save_dir_split}")

        # 4. Run the main loop
        for videos, fileid, st in tqdm.tqdm(dataloader, total=len(dataset)):
            if not videos: # Skip if video was bad
                print(f"Warning: Skipping empty video for fileid {fileid}")
                continue

            # This inner loop batches the clips *within* one video
            # 'args.batch_size' is the GPU batch size (e.g., 16 or 32)
            feats_per_video = []
            for j in range(0, len(videos), args.batch_size):
                chunk = videos[j : j + args.batch_size]
                feats = reader.get_feats(chunk).cpu().numpy()
                feats_per_video.append(feats)

            # Concatenate all features for this video
            feats = np.concatenate(feats_per_video, axis=0)

            # Create the final filename and save
            postfix = (f"_{st}" if st is not None else "") + f"_overlap-{args.overlap_size}"
            np.save(osp.join(save_dir_split, f"{fileid}{postfix}.npy"), feats)

        print(f"✅ Extraction for '{_m}' complete.")


if __name__ == "__main__":
    main()
