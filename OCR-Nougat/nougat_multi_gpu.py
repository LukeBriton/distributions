import os
import torch
import pypdfium2 as pdfium
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import NougatProcessor, VisionEncoderDecoderModel

def render_page(pdf, page_idx: int, scale: float = 2.0) -> Image.Image:
    page = pdf[page_idx]
    try:
        bmp = page.render(scale=scale)          # pypdfium2 render
        return bmp.to_pil()
    finally:
        page.close()

def main(pdf_path: str, out_path: str = "out.mmd", batch_size: int = 6, scale: float = 2.0):
    # Speed knobs (Ampere-friendly)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(mixed_precision="fp16")

    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "facebook/nougat-base",
        torch_dtype=torch.float16
    ).to(accelerator.device).eval()

    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)

    # Split pages across processes by striding (no duplicates, no padding)
    rank = accelerator.process_index
    world = accelerator.num_processes
    my_pages = list(range(rank, n_pages, world))

    local_results = []

    with torch.inference_mode():
        for i in range(0, len(my_pages), batch_size):
            batch_page_ids = my_pages[i:i + batch_size]
            images = [render_page(pdf, p, scale=scale) for p in batch_page_ids]

            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(accelerator.device, dtype=torch.float16)

            # Fast-ish decoding settings; tune max_new_tokens as needed
            gen_ids = model.generate(
                pixel_values,
                num_beams=1,
                do_sample=False,
                max_new_tokens=4096,
                use_cache=True,
            )

            texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
            local_results.extend(list(zip(batch_page_ids, texts)))

    pdf.close()

    # Gather all (page_idx, text) pairs to every process; we’ll write only on main
    # gather_object(local_results) already flattens across processes (it takes each rank’s list and returns one single list).
    # In Accelerate’s implementation, _gpu_gather_object() does all_gather_object(...) and then explicitly flattens one level.
    flat = gather_object(local_results)

    if accelerator.is_main_process:
        # No need for `flat = [item for sub in all_results for item in sub]` here.
        # If all_results is already a list of tuples (page_idx, text), then iterating for item in sub iterates over the tuple
        # → yields page_idx (an int) and text (a str), and then flat contains ints/strings → x[0] explodes with “int is not subscriptable”.
        flat.sort(key=lambda x: x[0])
        with open(out_path, "w", encoding="utf-8") as f:
            for page_idx, text in flat:
                f.write(f"\n\n%% --- page {page_idx+1} ---\n\n")
                f.write(text)
        accelerator.print(f"Wrote: {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="input PDF")
    ap.add_argument("--out", default="out.mmd")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()
    main(args.pdf, args.out, args.batch, args.scale)
