import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import NougatProcessor, VisionEncoderDecoderModel

from typing import Optional, List
import io
import fitz # pip install PyMuPDF
from pathlib import Path

def rasterize_paper(
    pdf: Path,
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil=False,
    pages=None,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """

    pillow_images = []
    if outpath is None:
        return_pil = True
    try:
        if isinstance(pdf, (str, Path)):
            pdf = fitz.open(pdf)
        if pages is None:
            pages = range(len(pdf))
        for i in pages:
            page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
            if return_pil:
                pillow_images.append(io.BytesIO(page_bytes))
            else:
                with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                    f.write(page_bytes)
    except Exception:
        pass
    if return_pil:
        return pillow_images

from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict

class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0

from PIL import Image

def main(pdf_path: str, out_path: str = "out.mmd", batch_size: int = 6):
    # Speed knobs (Ampere-friendly)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(mixed_precision="fp16")

    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "facebook/nougat-base",
        # `torch_dtype` is deprecated! Use `dtype` instead!
        dtype=torch.float16
    ).to(accelerator.device).eval()

    n_pages = len(fitz.open(pdf_path))

    # Split pages across processes by striding (no duplicates, no padding)
    rank = accelerator.process_index
    world = accelerator.num_processes
    my_pages = list(range(rank, n_pages, world))

    local_results = []

    with torch.inference_mode():
        for i in range(0, len(my_pages), batch_size):
            batch_page_ids = my_pages[i:i + batch_size]
            # images = rasterize_paper(pdf_path, return_pil=False, pages=batch_page_ids)
            # ValueError: Could not make a flat list of images from [<_io.BytesIO object at 0x7f663b374b30>, <_io.BytesIO object at 0x7f66240fd5d0>, <_io.BytesIO object at 0x7f666552dd00>]
            # NougatProcessor/NougatImageProcessor
            # images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]
            # images (ImageInput) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
            images = [Image.open(image_bytes) for image_bytes in rasterize_paper(pdf_path, return_pil=False, pages=batch_page_ids)]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(accelerator.device, dtype=torch.float16)

            # Fast-ish decoding settings; tune max_new_tokens as needed
            # If your stopping criteria depends on the scores input, make sure you pass return_dict_in_generate=True, output_scores=True to generate.
            # This feature is intended for advanced users.
            gen_ids = model.generate(
                pixel_values,
                num_beams=1,
                do_sample=False,
                min_length=1,
                max_new_tokens=4096,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
                use_cache=True,
            )
            # if return_dict_in_generate=False & output_scores=False, then gen_ids is just a tensor, use
            # texts = processor.batch_decode(gen_ids, skip_special_tokens=True) to decode
            texts = processor.batch_decode(gen_ids.sequences, skip_special_tokens=True)
            texts = processor.post_process_generation(texts, fix_markdown=False)
            local_results.extend(list(zip(batch_page_ids, texts)))

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
    args = ap.parse_args()
    main(args.pdf, args.out, args.batch)
