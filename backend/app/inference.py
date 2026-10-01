# ============================================================
# ONNX INFERENCE ENGINE — SegFormer-B2
# ============================================================

import os
from pathlib import Path
import onnxruntime as ort
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (matches training pipeline in segformer.ipynb)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ============================================================
# ONNX INFERENCE CLASS
# ============================================================

class ONNXInference:
    """
    ONNX inference engine for SegFormer-B2 field segmentation.

    The model outputs logits of shape (1, 2, H', W') where:
      - class 0 = background / boundary
      - class 1 = farmland field

    We apply softmax over the class axis and return the
    probability map for class 1 (field pixels).
    """

    def __init__(self, model_path, providers=None, tile_size=512, stride=256, blend_sigma=None):
        """
        Parameters
        ----------
        model_path   : Path to segformer_farmland.onnx
        providers    : ONNX Runtime execution providers
        tile_size    : Patch size for sliding-window inference (must match export dummy size)
        stride       : Step size between overlapping patches
        blend_sigma  : Gaussian sigma for blending weight map
        """

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.tile_size  = tile_size
        self.stride     = stride
        self.blend_sigma = blend_sigma if blend_sigma is not None else tile_size / 8.0

        logger.info(f"Loading SegFormer-B2 ONNX model from: {model_path}")

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)

            logger.info(f"Available providers : {ort.get_available_providers()}")
            logger.info(f"Using providers     : {self.session.get_providers()}")

            self.input_name  = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            logger.info(f"Input  name/shape : {self.input_name}  {self.session.get_inputs()[0].shape}")
            logger.info(f"Output name/shape : {self.output_name} {self.session.get_outputs()[0].shape}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    # ----------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------
    def preprocess(self, image, target_h=None, target_w=None):
        """
        Convert a BGR OpenCV image to a normalised float32 tensor
        using ImageNet mean/std (matching the val_transform in the notebook).

        Returns shape (1, 3, target_h, target_w).
        """
        target_h = target_h or self.tile_size
        target_w = target_w or self.tile_size

        resized = cv2.resize(image, (int(target_w), int(target_h)),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalise: /255 then (x - mean) / std
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - _MEAN) / _STD

        tensor = np.transpose(arr, (2, 0, 1))          # HWC → CHW
        tensor = np.expand_dims(tensor, 0).astype(np.float32)  # → (1,3,H,W)
        return tensor

    # ----------------------------------------------------------
    # Postprocessing
    # ----------------------------------------------------------
    def postprocess(self, logits, original_height, original_width):
        """
        Convert raw logits (1, 2, H', W') to a field probability map (H, W).

        Steps:
          1. Softmax over class axis (axis 1)
          2. Take class-1 (field) probability channel
          3. Resize back to original image dimensions
        """
        # logits: (1, 2, H', W') → (2, H', W')
        logits = logits.squeeze(0)

        # Softmax over classes
        exp_l = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp_l / exp_l.sum(axis=0, keepdims=True)   # (2, H', W')

        # Field probability channel (class 1)
        field_prob = probs[1]  # (H', W')

        # Resize to original image size
        field_prob_resized = cv2.resize(
            field_prob.astype(np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        return np.clip(field_prob_resized, 0.0, 1.0)

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _get_tile_positions(self, dimension):
        if dimension <= self.tile_size:
            return [0]
        positions = list(range(0, dimension - self.tile_size + 1, self.stride))
        if positions[-1] != dimension - self.tile_size:
            positions.append(dimension - self.tile_size)
        return positions

    def _pad_patch(self, patch):
        patch_h, patch_w = patch.shape[:2]
        pad_h = max(0, self.tile_size - patch_h)
        pad_w = max(0, self.tile_size - patch_w)
        if pad_h == 0 and pad_w == 0:
            return patch, 0, 0
        padded = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        return padded, pad_h, pad_w

    def _create_weight_map(self):
        g1d    = cv2.getGaussianKernel(self.tile_size, self.blend_sigma)
        weight = (g1d @ g1d.T).astype(np.float32)
        weight /= weight.max()
        return weight

    def _save_debug_image(self, output_dir, filename, image):
        if output_dir is None:
            return None
        output_path = Path(output_dir) / filename
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image, 0.0, 1.0)
                image = (image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), image)
        return str(output_path)

    def _infer_patch(self, patch):
        """Run inference on a single BGR patch and return the field probability map."""
        input_tensor = self.preprocess(patch)
        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        return self.postprocess(logits, patch.shape[0], patch.shape[1])

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def predict(self, image, debug_output_dir=None):
        """
        Run sliding-window inference on a full image.

        Returns
        -------
        mask      : float32 (H, W) field probability map in [0, 1]
        debug_info: dict with patch metadata
        """
        mask, debug_info = self.predict_with_overlap(image, debug_output_dir=debug_output_dir)

        if debug_output_dir is not None:
            self._save_debug_image(debug_output_dir, 'debug_prediction.png', mask)

        return mask, debug_info

    def predict_with_overlap(self, image, debug_output_dir=None):
        image_h, image_w = image.shape[:2]

        positions_y  = self._get_tile_positions(image_h)
        positions_x  = self._get_tile_positions(image_w)

        merged     = np.zeros((image_h, image_w), dtype=np.float32)
        normalizer = np.zeros((image_h, image_w), dtype=np.float32)
        weight_map = self._create_weight_map()

        patch_files = []
        patch_idx   = 0

        for y in positions_y:
            for x in positions_x:
                patch             = image[y:y + self.tile_size, x:x + self.tile_size]
                patched, pad_h, pad_w = self._pad_patch(patch)

                prediction = self._infer_patch(patched)

                if pad_h > 0 or pad_w > 0:
                    prediction = prediction[:patch.shape[0], :patch.shape[1]]

                patch_h, patch_w = prediction.shape[:2]
                weight = weight_map[:patch_h, :patch_w]

                merged    [y:y + patch_h, x:x + patch_w] += prediction * weight
                normalizer[y:y + patch_h, x:x + patch_w] += weight

                if debug_output_dir is not None and patch_idx < 12:
                    f = self._save_debug_image(
                        debug_output_dir,
                        f'debug_patch_{patch_idx + 1}.png',
                        prediction
                    )
                    if f:
                        patch_files.append(f)

                patch_idx += 1

        valid = normalizer > 1e-6
        merged[valid]  /= normalizer[valid]
        merged[~valid]  = 0.0

        debug_info = {'patch_files': patch_files, 'num_patches': patch_idx}
        return merged, debug_info
