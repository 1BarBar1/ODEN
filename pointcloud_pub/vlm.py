from platform import processor

import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from ultralytics import YOLOWorld, SAM, YOLO
from PIL import Image
import cv2
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# import matplotlib.pyplot as plt
import time


class Clipseg:
    def __init__(self, prompts=["Blocks"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)

        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

        self.model.to(self.device)
        # .half() # Move to GPU and use FP16
        self.model.eval()

        self.prompts = prompts

    def get_segmentation(self, input):
        self.input = input[:, :, ::-1]
        self.input = Image.fromarray(self.input.astype("uint8"), mode="RGB")
        self.image = self.input
        vision_size = self.model.config.vision_config.image_size
        self.processor.image_processor.size = {
            "height": vision_size,
            "width": vision_size,
        }
        self.processor.image_processor.do_resize = True
        self.processor.image_processor.do_center_crop = False
        inputs = self.processor(
            text=self.prompts,
            images=[self.input] * len(self.prompts),
            return_tensors="pt",
        )

        # inputs = {k: v.to(self.device) for k,v in inputs.items()}
        # match model precision
        # inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # shape: (num_prompts, H, W)
        mask = torch.sigmoid(logits)
        mask = mask.unsqueeze(0).unsqueeze(0)  # NCHW
        mask = F.interpolate(mask, (logits.size(dim=0), 480, 640), mode="nearest")

        return mask[0, 0].detach().cpu().numpy(), logits.cpu()


class YoloSamCombo:
    def __init__(self, prompts=["hammer"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)

        self.detector = YOLOWorld(
            "/home/rosdev/ros2_ws/src/R7018E/pointcloud_pub/models/yolov8s-worldv2.pt"
        )
        self.detector.set_classes(prompts)
        self.segmenter = SAM(
            "/home/rosdev/ros2_ws/src/R7018E/pointcloud_pub/models/mobile_sam.pt"
        )

    def get_segmentation(self, input):
        det_results = self.detector.predict(input, conf=0.04, device=self.device)
        boxes = det_results[0].boxes.xyxy
        print(f"Detected {len(boxes)} objects with YOLOv8!")
        if len(boxes) > 0:
            print(f"Found {len(boxes)} objects! Generating crisp masks...")

            # Step B: Feed those exact boxes into SAM
            seg_results = self.segmenter.predict(input, bboxes=boxes, device="cpu")

            if seg_results[0].masks is not None:
                # 1. Get the original camera image dimensions (Height, Width)
                orig_h, orig_w = seg_results[0].orig_shape

                # 2. Extract the raw mask tensors and move them to CPU -> NumPy
                # Shape is (N, H, W) where N is number of detected objects
                raw_masks = seg_results[0].masks.data.cpu().numpy()

                # 3. Grab the mask for the first detected object (e.g., the hammer)
                first_tool_mask = raw_masks[0]

                # 4. Resize the mask back to the original camera resolution
                # CRITICAL: Use INTER_NEAREST so the edges stay strictly 0 or 1, without blurry gradients
                full_res_mask = cv2.resize(
                    first_tool_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                # 5. Convert to a strict Boolean array for indexing
                boolean_mask = full_res_mask > 0.5

                print(
                    f"Mask extracted! Shape: {boolean_mask.shape}, Type: {boolean_mask.dtype}"
                )
                # Output will be: Shape: (480, 640), Type: bool
                return boolean_mask
            else:
                print("No masks found for the detected objects.")
                return []  # Return the original image if no tools are found

        else:
            print("No tools found. Try lowering the confidence threshold.")
            return []  # Return the original image if no tools are found


class Yolo26e:
    def __init__(self, prompts=["hammer"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)

        # Load the lightweight YOLOE-26 segmentation model
        self.model = YOLO("yoloe-26s-seg.pt")

        # Set your text prompts just like YOLO-World
        self.model.set_classes(["hammer", "screwdriver"])

        # Predict on the image (returns bounding boxes AND masks simultaneously)

    def get_segmentation(self, input):
        results = self.model.predict(input, conf=0.01)

        # Extract the binary masks directly for your 3D point cloud
        if results[0].masks is not None:
            # Get the raw mask tensors and convert to a NumPy array
            raw_masks = results[0].masks.data.cpu().numpy()

            print(f"Success! Found {len(raw_masks)} tool masks directly from YOLOE-26.")
            # Proceed to resize and project onto your ROS 2 PointCloud...
            return raw_masks[0]  # Return the first detected tool mask
        else:
            print("No tools found.")
            return []  # Return the original image if no tools are found


class YoloVlmSam:
    def __init__(self, prompts=["hammer"]):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)

        self.vlm_processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct"
        )
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).to(device)

        # Load the lightweight YOLOE-26 segmentation model
        self.yolo_model = YOLO("yolo26n.pt")

        self.segmenter = SAM(
            "/home/rosdev/ros2_ws/src/R7018E/pointcloud_pub/models/mobile_sam.pt"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Please give the index objects in the image that matches the following description: "
                        + prompts,
                    },
                ],
            },
        ]
        self.prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        # Predict on the image (returns bounding boxes AND masks simultaneously)

    def get_segmentation(self, input):

        det_results = self.yolo_model.predict(input, conf=0.01, device=self.device)
        boxes = det_results[0].boxes.xyxy
        print(f"Detected {len(boxes)} objects with YOLOv8!")
        vlm_inputs = self.vlm_processor(
            text=self.prompt, images=[boxes[0]], return_tensors="pt"
        )
        vlm_inputs = vlm_inputs.to(self.device)
        generated_ids = self.vlm_model.generate(**vlm_inputs, max_new_tokens=500)
        generated_texts = self.vlm_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        print(generated_texts[0])

        results = self.model.predict(input, conf=0.01)

        # Extract the binary masks directly for your 3D point cloud
        if results[0].masks is not None:
            # Get the raw mask tensors and convert to a NumPy array
            raw_masks = results[0].masks.data.cpu().numpy()

            print(f"Success! Found {len(raw_masks)} tool masks directly from YOLOE-26.")
            # Proceed to resize and project onto your ROS 2 PointCloud...
            return raw_masks[0]  # Return the first detected tool mask
        else:
            print("No tools found.")
            return []  # Return the original image if no tools are found
