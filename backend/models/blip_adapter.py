"""BLIP adapter — supports image captioning and VQA.

Model: Salesforce/blip-image-captioning-base  (captioning)
       Salesforce/blip-vqa-base               (VQA)

Both are small enough to run on CPU for a demo; GPU strongly recommended.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor

from backend.models.base import BaseModelAdapter

logger = logging.getLogger(__name__)

CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
VQA_MODEL_ID = "Salesforce/blip-vqa-base"


class BLIPAdapter(BaseModelAdapter):
    """BLIP image captioning + VQA adapter."""

    name = "blip-base"
    supports_captioning = True
    supports_vqa = True
    supports_embeddings = False

    def load(self) -> None:
        logger.info("Loading BLIP captioning model (%s) …", CAPTION_MODEL_ID)
        self._caption_processor = BlipProcessor.from_pretrained(
            CAPTION_MODEL_ID, cache_dir=self.cache_dir
        )
        self._caption_model = BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_ID, cache_dir=self.cache_dir
        ).to(self.device)

        logger.info("Loading BLIP VQA model (%s) …", VQA_MODEL_ID)
        self._vqa_processor = BlipProcessor.from_pretrained(
            VQA_MODEL_ID, cache_dir=self.cache_dir
        )
        self._vqa_model = BlipForQuestionAnswering.from_pretrained(
            VQA_MODEL_ID, cache_dir=self.cache_dir
        ).to(self.device)

        self._caption_model.eval()
        self._vqa_model.eval()
        logger.info("BLIP models loaded.")

    def caption(self, image: Image.Image) -> str:
        """Generate a caption for the given PIL image."""
        self.ensure_loaded()
        inputs = self._caption_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._caption_model.generate(**inputs, max_new_tokens=100)
        return self._caption_processor.decode(out[0], skip_special_tokens=True)

    def answer(self, image: Image.Image, question: str) -> str:
        """Answer a visual question about the given image."""
        self.ensure_loaded()
        inputs = self._vqa_processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._vqa_model.generate(**inputs, max_new_tokens=50)
        return self._vqa_processor.decode(out[0], skip_special_tokens=True)
