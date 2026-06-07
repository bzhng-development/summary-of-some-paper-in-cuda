"""OCR pipeline: fill in full-text markdown for papers HF can't render.

Three stages:

1. ``find_missing`` — probe ``huggingface.co/papers/{id}.md`` for every arxiv-pattern
   paper in Neon and bucket each as ``hf_ok`` / ``html_only`` / ``ocr_needed``.
2. ``chandra_ocr`` (self-contained PEP-723 script, runs on the GPU box) — OCR the
   ``ocr_needed`` PDFs with ``datalab-to/chandra-ocr-2`` served by vLLM.
3. ``ingest_ocr`` — upsert the OCR'd markdown back into Neon (new ``markdown`` column).
"""
