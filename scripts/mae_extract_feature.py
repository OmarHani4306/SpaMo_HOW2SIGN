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
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.helpers import sliding_window_for_list, read_video, get_img_list

# --- GLOBAL SETTINGS --------------------------------------------------------
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
class VideoMAEFeatureReader(object):
    """Holds model + GPU inference."""
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
        inputs = self.image_processor(images=video_batch, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        feats = outputs[self.nth_layer][:, 0]     # CLS token
        return feats


# ----------------------------------------------------------------------------
class VideoDataset(Dataset):
    """CPU side: load frames, crop them into windows."""
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        self.data = np.load(
            osp.join(args.anno_root, f"{mode}_info.npy"),
            allow_pickle=True
        ).item()

        self.num_videos = len(self.data)
        self.ds_name = osp.split(args.anno_root)[-1]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        entry = self.data[idx]
        fname, fileid = entry["folder"], entry["fileid"]
        st_return_str = None # This will be the string for the filename
        videos = []

        if self.ds_name in ["Phoenix1T", "CSL-Daily"]:
            image_list = get_img_list(self.ds_name, self.args.video_root, fname)
            image_list = image_list + [image_list[-1]] * max(0, 16 - len(image_list))
            clips = sliding_window_for_list(image_list, 16, self.args.overlap_size)

            for clip in clips:
                pil_frames = []
                for path in clip:
                    try:
                        img = Image.open(path).convert("RGB")
                        pil_frames.append(img.copy())
                        img.close()
                    except Exception as e:
                        print(f"Warning: Failed to load image {path}: {e}")
                        continue # Skip corrupted images
                if pil_frames:
                    videos.append(pil_frames)

        elif self.ds_name == "How2Sign":
            # --- START OF FIX ---
            
            # 1. Get the raw start/end time values
            start_time_val = entry["original_info"]["START_REALIGNED"]
            end_time_val = entry["original_info"]["END_REALIGNED"]

            # 2. Convert them to float, handling None, NaN, or empty strings
            try:
                start_time_float = float(start_time_val)
            except (ValueError, TypeError):
                start_time_float = None # Pass None if conversion fails

            try:
                end_time_float = float(end_time_val)
            except (ValueError, TypeError):
                end_time_float = None

            # 3. Pass the FLOATS (or Nones) to read_video
            frames = read_video(fname, start_time=start_time_float, end_time=end_time_float)
            
            # 4. Create the string version for the return value *after* reading
            st_return_str = str(start_time_float) if start_time_float is not None else "None"
            
            # --- END OF FIX ---

            if not frames:
                return ([], fileid, st_return_str)


            frames = frames + [frames[-1]] * max(0, 16 - len(frames))
            videos = sliding_window_for_list(frames, 16, self.args.overlap_size)
            #  Convert PIL images → tensors
            videos = [[self.transform(f) for f in clip] for clip in videos]
        else:
            raise NotImplementedError

        return videos, fileid, st_return_str


# ----------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', required=True)
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--mode', nargs='+', type=str)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser


# ----------------------------------------------------------------------------
def main():
    args = get_parser().parse_args()

    reader = VideoMAEFeatureReader(
        args.model_name,
        args.device,
        args.overlap_size,
        args.nth_layer,
        args.cache_dir
    )

    mode = ["test"]  # requested
    for m in mode:
        ds_name = osp.split(args.anno_root)[-1]
        out_folder = f"mae_feat_{ds_name}"
        os.makedirs(osp.join(args.save_dir, out_folder, m), exist_ok=True)

        # Adjust internal mode name for datasets (How2Sign etc.)
        if ds_name == "How2Sign":   _m = "val" if m == "dev" else m
        elif ds_name == "NIASL2021": _m = "validation" if m == "dev" else m
        else:                        _m = m

        dataset = VideoDataset(args, _m)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        print(f"Extracting '{_m}' using {args.num_workers} workers...")

        for batch in tqdm.tqdm(dataloader, total=len(dataset)):
            videos, fileid, st = batch[0]
            if not videos:
                continue

            feats_per_video = []
            for j in range(0, len(videos), args.batch_size):
                chunk = videos[j : j + args.batch_size]
                feats = reader.get_feats(chunk).cpu().numpy()
                feats_per_video.append(feats)

            feats = np.concatenate(feats_per_video, axis=0)

            save_path = osp.join(args.save_dir, out_folder, _m)
            postfix = (f"_{st}" if st is not None else "") + f"_overlap-{args.overlap_size}"
            np.save(osp.join(save_path, f"{fileid}{postfix}.npy"), feats)

        print(f"✅ Extraction for '{_m}' complete.")


if __name__ == "__main__":
    main()
