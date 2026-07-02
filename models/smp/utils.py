from collections import defaultdict
import os
import torch
from typing import Optional, Any
from PIL import Image

from tqdm import tqdm

from models.smp.exceptions import NoValidAutobatchConfigException

AUTOBATCH_SIZES: tuple = (1, 2, 4, 8, 16, 32, 64)
VALID_EXTENSIONS: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
IMAGE_RESIZE_VALUES: tuple = (224, 384, 512, 640, 1024)

def calculate_model_size(model: torch.nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_gb = total_size_bytes / (1024 ** 3)  
    return total_size_gb

def _extract_first_tensor(obj):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    for attr in ["logits", "out", "preds", "class_queries_logits", "masks_queries_logits", "last_hidden_state"]:
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            if torch.is_tensor(v) or (isinstance(v, (list, tuple, dict)) and _extract_first_tensor(v) is not None):
                return _extract_first_tensor(v)
    return None

def _tensorlist_to_hwc_numpy_list(tensor: torch.Tensor):
    # tensor: (B, C, H, W) o (C, H, W)
    if tensor.dim() == 4:
        # Batch
        return [img.permute(1, 2, 0).cpu().numpy() for img in tensor]
    elif tensor.dim() == 3:
        # Single image
        return [tensor.permute(1, 2, 0).cpu().numpy()]
    else:
        raise ValueError("3D or 4D tensor expected")

def profile_memory(
    img: Any,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    processor = None,
    verbose: bool = True
) -> float:

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gb = 1 << 30

    model.to(device)
    model.train()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.zero_grad(set_to_none=True)
    proc_out = None
    
    try:
        inputs_for_processor = img

        if processor is not None:
            args = {
                "images": inputs_for_processor,
                "do_resize": False,
                "do_normalize": True,
                "do_rescale": False,
                "return_tensors": "pt",
            }

            if torch.is_tensor(inputs_for_processor):
                images_list = _tensorlist_to_hwc_numpy_list(inputs_for_processor)
                args["images"] = images_list
                            
            proc_out = processor(**args)
                
        img_device = {}
        if proc_out is not None:
            for k, v in proc_out.items():
                if torch.is_tensor(v):
                    img_device[k] = v.to(device)
                elif isinstance(v, (list, tuple)):
                    new_list = []
                    for elem in v:
                        if torch.is_tensor(elem):
                            new_list.append(elem.to(device))
                        else:
                            new_list.append(elem)
                    img_device[k] = new_list
                else:
                    img_device[k] = v
            outputs = model(**img_device)
        else:
            if torch.is_tensor(img):
                img = img.to(device)
            outputs = model(img)

    except Exception as e:
        if verbose:
            err_name = type(e).__name__
            if "out of memory" in str(e).lower():
                try:
                    batch = img.shape[0]
                except Exception:
                    batch = "?"
                print(f"CUDA OOM in forward (batch {batch})")
            else:
                print(f"Forward call failed ({err_name}): {e}")
        raise e
    
    loss = None
    
    # HF
    if hasattr(outputs, "loss") and getattr(outputs, "loss") is not None:
        loss = outputs.loss
        if verbose:
            print("Using outputs.loss from model.")
    else:
        # SMP
        first_tensor = _extract_first_tensor(outputs)
        if first_tensor is not None and torch.is_tensor(first_tensor):
            loss = first_tensor.mean()
            if verbose:
                print("Using mean(first_tensor) as fallback loss.")
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if verbose:
                print("No tensor found in outputs and no labels/loss_fn provided: using dummy scalar loss.")

    try:
        loss.backward()
    except Exception as e:
        try:
            loss = loss.float()
            loss.backward()
        except Exception:
            if verbose:
                print("Backward failed:", e)
            raise

    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / gb
        
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    return float(peak)


def autobatch(
    model: torch.nn.Module,
    imgsz=224,
    fraction=0.6,
    processor=None,
) -> int:
    device = next(model.parameters()).device
        
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    except ValueError as e:
        raise RuntimeError(
            "CUDA operations attempted on CPU. Please run the training on a GPU-enabled environment."
        ) from e
    
    if device.type == 'cuda':
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        
        d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"  
        properties = torch.cuda.get_device_properties(device)  # device properties
        t = properties.total_memory / gb  # GiB total
        r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
        a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
        f = t - (r + a)  # GiB free
        model_size = calculate_model_size(model)
        usable_mem = f * fraction
        
        print(f"{d} device properties (GiB): ")
        print(f"    Total memory: {t:.2f}")
        print(f"    Reserved memory: {r:.2f}")
        print(f"    Allocated memory: {a:.2f}")
        print(f"    Free memory: {f:.2f}")
        print(f"    Model size: {model_size:.2f}")
        print(f"    Usable memory: {usable_mem:.2f}")
        
        if f < 0 or usable_mem < 0:
            raise NoValidAutobatchConfigException(usable_mem)
            
        if model_size > usable_mem:
            raise NoValidAutobatchConfigException(usable_mem, f"Model size ({model_size:.2f} GB) exceeds usable memory ({usable_mem:.2f} GB).")
    else: 
        usable_mem = None 
        
    batch_sizes = sorted(AUTOBATCH_SIZES, reverse=True)

    best_batch = None
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        imgs = torch.empty(bs, 3, imgsz, imgsz, device=device)

        try:
            mem_used = profile_memory(imgs, model, device, processor=processor)
            print(f"Testing batch_size={bs}, mem={mem_used:.2f}GB")
        except Exception as e:
            error_msg = f"  Error profiling batch_size={bs}: "
            if isinstance(e, (torch.cuda.OutOfMemoryError, RuntimeError)) and "out of memory" in str(e).lower():
                error_msg += " CUDA OOM: Out of memory in GPU."
            continue
                        
        if usable_mem and mem_used <= usable_mem:
            best_batch = bs
            break   
        
    if best_batch is None:
        raise NoValidAutobatchConfigException(usable_mem)

    print(f"Best batch size found:{best_batch}")
    return best_batch 

def image_sizes(directory): 
        """
        Returns the sizes of the images in the directory.
        Also calculates the average width and height.
        """
        images_sizes = defaultdict(int)
        total_width = 0
        total_height = 0

        for fname in tqdm(os.listdir(directory), desc="Reading files"):
            fpath = os.path.join(directory, fname)
            if fname.lower().endswith(tuple(VALID_EXTENSIONS)):   
                with Image.open(fpath) as img:
                    width, height = img.size
                    total_width += width
                    total_height += height
                    images_sizes[(width, height)] += 1

        mode_height, mode_width = max(set(images_sizes), key=list(images_sizes.values()).count)
        sorted_sizes = sorted(images_sizes.items(), key=lambda item: item[1], reverse=True)
        images_sizes = dict(sorted_sizes)
        
        for size, count in images_sizes.items():
            width, height = size

        avg_width = round(total_width / len(images_sizes))
        avg_height = round(total_height / len(images_sizes))
        print(f"Average image size: {avg_height}x{avg_width}")
        print(f"Image size mode: {mode_height}x{mode_width}")

        return mode_height, mode_width

def calculate_closest_resize(mode_height, mode_width, stride=32, max_padding=16):

        valid_sizes = [s for s in IMAGE_RESIZE_VALUES if s % stride == 0]

        original_mode_height = mode_height
        original_mode_width = mode_width

        if mode_height != mode_width:
            max_side = max(mode_height, mode_width)
            print(f"Making size square by using {max_side}px height and {max_side}px width.")
            mode_height = mode_width = max_side

        closest_height = min(valid_sizes, key=lambda x: abs(x - mode_height))
        diff_height = abs(closest_height - mode_height)
        if diff_height <= max_padding:
            if closest_height != mode_height:
                print(f"Dimensions {mode_height}px adjusted to {closest_height}px (compatible with stride {stride}).")
            mode_height = mode_width = closest_height
        else:
            lower_valid = [s for s in valid_sizes if s <= mode_height]
            new_height = max(lower_valid) if lower_valid else min(valid_sizes)
            if new_height != mode_height:
                print(f"Dimensions {mode_height}px adjusted down to {new_height}px (compatible with stride {stride}).")
            mode_height = mode_width = new_height

        if original_mode_height != mode_height or original_mode_width != mode_height:
            print(f"Images will be resized to {mode_height}px height and {mode_width}px width.")
            
        return mode_height
