import gradio as gr
from PIL import Image
import numpy as np



# =============================================
print("==> Initializing")
from box import Box

args = Box()
# args.checkpoint_path = 'ztmp/checkpoint_1ep.pth.tar'
# args.checkpoint_path = 'ztmp/checkpoint_10ep.pth.tar'
# args.checkpoint_path = 'ztmp/27kpng_model_best.pth.tar'
# args.checkpoint_path = 'ztmp/checkpoint_finetune_93ep.pth.tar'
# args.checkpoint_path = 'ztmp/checkpoint_finetune_99ep.pth.tar'
# args.checkpoint_path = 'ztmp/checkpoint_finetune2_93ep.pth.tar'
args.checkpoint_path = 'ztmp/checkpoint_finetune2_100ep.pth.tar'
args.device = 'cpu'


import torch
import scripts.models as archs

checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
model = archs.__dict__[checkpoint['arch']]()
model.load_state_dict(checkpoint['state_dict'])

model.eval()


import torchvision.transforms as transforms
trans = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )


def split_segments(
    img, 
    patch_size = 256,
    stride = 192,
):
    width, height = img.size

    assert (patch_size - stride) % 2 == 0
    pad = (patch_size - stride) // 2 

    segments = []
    positions = []
    for sx in range(-pad, width+pad, stride):
        for sy in range(-pad, height+pad, stride):
            ex, ey = sx + patch_size, sy + patch_size
            patch = img.crop((sx, sy, ex, ey))
            positions.append((sx, sy, ex, ey))
            segments.append(patch)

    return segments, positions


def reconstruct_image(
    segments,
    positions,
    width,
    height,
    patch_size=256,
    stride=192,
    mode='RGB',
):
    # assert len(segments) == len(positions)
    assert (patch_size - stride) % 2 == 0
    pad = (patch_size - stride) // 2 

    img = Image.new(mode, (width, height))

    for i, (pos, seg) in enumerate(zip(positions, segments)):
        x1, x2, y1, y2 = pos
        x1 = x1 + pad
        y1 = y1 - pad
        x2 = x2 + pad
        y2 = y2 - pad
        seg = seg.crop((pad, pad, patch_size - pad, patch_size - pad))
        img.paste(seg, (x1, x2, y1, y2))

    return img


import cv2

def get_filtered_mask(mask):
    mask_gray = cv2.medianBlur(mask, 21) 
    mask_gray = (mask_gray > 150).astype("uint8") * 255

    mask2 = cv2.dilate(mask_gray, np.ones((40, 40), np.uint8))

    mask3 = cv2.bitwise_and(mask, mask2)
    mask3 = cv2.bitwise_and(mask, mask2)
    mask3 = mask3 > 200
    mask3 = mask3.astype("uint8")
    mask3 = mask3 * 255

    return mask3


def fill_img_with_mask(img, pred, mask):
    img3 = np.zeros_like(img)
    img3[mask == 255] = pred[mask == 255]
    img3[mask != 255] = img[mask != 255]
    
    return img3


# =============================================


def process_images(img, progress=gr.Progress()):
    print("Processing image")
    # processed_image1 = image.convert("L")

    width, height = img.size
    segments, positions = split_segments(img)

    total_segments = len(segments)

    refine_segments = []
    reconstructed_segments = []
    masks = []
    for i, seg in enumerate(segments):
        seg = trans(seg)
        batch = seg.unsqueeze(0)

        [refine, reconstructed_image], reconstructed_mask, reconstructed_vm = model(batch)

        out = refine.detach().cpu().numpy()[0]
        out = np.clip(out, 0, 1)
        out = out.transpose(1, 2, 0)
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        refine_segments.append(out)

        out = reconstructed_image.detach().cpu().numpy()[0]
        out = np.clip(out, 0, 1)
        out = out.transpose(1, 2, 0)
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        reconstructed_segments.append(out)

        mask = reconstructed_mask.detach().cpu().numpy()[0][0]
        mask = np.array(mask)
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        masks.append(mask)

        # update progress here
        # print(f"progress: {i+1}/{len(segments)}")
        progress((i + 1, total_segments), desc=f"Processing segment {i+1}/{total_segments}")

    img_refine = reconstruct_image(refine_segments, positions, width, height)
    img_reconstructed = reconstruct_image(reconstructed_segments, positions, width, height)
    mask2 = reconstruct_image(masks, positions, width, height, mode="L")


    mask3 = get_filtered_mask(mask2)

    img3 = fill_img_with_mask(img, img_reconstructed, mask3)


    return img3, mask3, img_refine, img_reconstructed, mask2


# Create the Gradio interface
interface = gr.Interface(
    fn=process_images,  # The inference function
    inputs=[
        gr.Image(type="pil", label="Input Image"),  # First image input
    ],
    outputs=[
        gr.Image(type="pil", label="Filtered image"),  # First output image
        gr.Image(type="pil", label="Filtered mask"),  # First output image
        gr.Image(type="pil", label="Refined"),  # First output image
        gr.Image(type="pil", label="Reconstructe"),  # Second output image
        gr.Image(type="pil", label="Mask")   # Third output image
    ],
    title="Watermark removal",
    description="Upload image, and the model will perform inference on them."
)

# Launch the interface
interface.launch(share=True)
