import re
from pathlib import Path

import pypdfium2 as pdfium
import torch
from accelerate import Accelerator
from transformers import NougatProcessor, VisionEncoderDecoderModel


def pdf_page_to_pil(doc: pdfium.PdfDocument, page_idx: int, dpi: int = 200):
    # pypdfium2 renders per-page in recent versions
    page = doc[page_idx]
    scale = dpi / 72.0  # PDF points are 72 dpi baseline
    bitmap = page.render(scale=scale)
    return bitmap.to_pil().convert("RGB")


def main(pdf_path: str):
    accelerator = Accelerator()
    device = accelerator.device

    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base").to(device)

    doc = pdfium.PdfDocument(pdf_path)

    outputs_all = []
    with torch.inference_mode():
        for i in range(len(doc)):
            image = pdf_page_to_pil(doc, i, dpi=200)
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

            # This matches the official inference pattern from the Transformers docs
            outputs = model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=4096,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
            )
            seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            seq = processor.post_process_generation(seq, fix_markdown=False)

            # Optional: add a page separator
            outputs_all.append(seq.strip())

    text = "\n\n---\n\n".join(outputs_all)
    # tiny cleanup: collapse 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    print(text)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
