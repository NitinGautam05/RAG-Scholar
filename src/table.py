# from pdf2image import convert_from_path
# import os
# import torch
# from config import Config
# from transformers import AutoImageProcessor, TableTransformerForObjectDetection
# from PIL import Image, ImageDraw


# class TableExtractor:
#     def __init__(self, pdf_path=Config.DATA_PATH, output_dir="D:/rag-chatbot/data"):
#         self.pdf_path = pdf_path
#         self.output_dir = output_dir
#         self.table_output_dir = os.path.join(output_dir, "tables")
#         os.makedirs(self.table_output_dir, exist_ok=True)

#         self.dpi = 300 
#         self.confidence_threshold = 0.99  
#         self.padding_factor = 0.05 

#         self.images = []
#         self.results = []
#         self.table_images = []

#         self.image_processor = None
#         self.detection_model = None

#     def convert_pdf_to_images(self):
#         pages_dir = os.path.join(self.output_dir, "pages")
#         os.makedirs(pages_dir, exist_ok=True)

#         try:
#             self.images = convert_from_path(self.pdf_path, dpi=self.dpi)
#             print(f"Successfully converted PDF to {len(self.images)} images.")
#         except Exception as e:
#             print(f"Error converting PDF: {e}")
#             return

#         for i, image in enumerate(self.images):
#             image_path = os.path.join(pages_dir, f"page_{i+1}.png")
#             image.save(image_path, "PNG")
#             print(f"Saved image: {image_path}")

#     def load_detection_model(self):
#             self.image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
#             self.detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
#             self.detection_model.eval()

#     def detect_tables(self):
#         pages_dir = os.path.join(self.output_dir, "pages")
#         image_paths = [os.path.join(pages_dir, f) for f in os.listdir(pages_dir) if f.startswith("page_") and f.endswith(".png")]
#         image_paths.sort()
#         self.images = [Image.open(path) for path in image_paths]

#         print(f"Loaded {len(self.images)} images for processing.")
#         self.results = []

#         for i, image in enumerate(self.images):
#             print(f"Processing page {i+1}...")
#             inputs = self.image_processor(images=image, return_tensors="pt")

#             with torch.no_grad():
#                 outputs = self.detection_model(**inputs)

#             target_sizes = torch.tensor([image.size[::-1]])  # [height, width]
#             result = self.image_processor.post_process_object_detection(
#                 outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
#             )[0]

#             valid_boxes = []
#             for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
#                 x1, y1, x2, y2 = box.tolist()
#                 width, height = x2 - x1, y2 - y1

#                 min_width = image.width * 0.1  
#                 min_height = image.height * 0.05

#                 if width >= min_width and height >= min_height:
#                     valid_boxes.append((box, score, label))

#             # If no tables are detected, try lowering confidence threshold dynamically
#             if not valid_boxes and self.confidence_threshold > 0.99:
#                 print(f"No tables detected on page {i+1}, reducing confidence threshold...")
#                 self.confidence_threshold -= 0.02  # Reduce threshold slightly and retry
#                 return self.detect_tables()

#             self.results.append(valid_boxes)

#             # Draw bounding boxes for debugging
#             draw = ImageDraw.Draw(image)
#             padding = 20

#             for box, _, _ in valid_boxes:
#                 x, y, x2, y2 = box.tolist()

#                 x_padded = max(0, x - padding)
#                 y_padded = max(0, y - padding)
#                 x2_padded = min(image.width, x2 + padding)
#                 y2_padded = min(image.height, y2 + padding)
#                 draw.rectangle([x_padded, y_padded, x2_padded, y2_padded], outline="red", width=3)


#     def crop_tables(self):
#         """Crop detected tables and save them."""
#         if not self.results:
#             print("No tables detected, skipping cropping.")
#             return

#         self.table_images = []
#         for i, (image, result) in enumerate(zip(self.images, self.results)):
#             if not result:
#                 print(f"Page {i+1}: No tables to crop.")
#                 continue

#             for j, (box, _, _) in enumerate(result):
#                 x, y, x2, y2 = box.tolist()

#                 # Adaptive padding based on table size
#                 padding_x = (x2 - x) * self.padding_factor
#                 padding_y = (y2 - y) * self.padding_factor

#                 x, y = max(0, x - padding_x), max(0, y - padding_y)
#                 x2, y2 = min(image.width, x2 + padding_x), min(image.height, y2 + padding_y)

#                 table_image = image.crop((x, y, x2, y2))
#                 table_path = os.path.join(self.table_output_dir, f"page_{i+1}_table_{j+1}.png")
#                 table_image.save(table_path)
#                 self.table_images.append(table_image)
#                 print(f"Saved cropped table: {table_path}")

#     def run(self):
#         """Execute the full table extraction pipeline."""
#         self.convert_pdf_to_images()
#         self.load_detection_model()
#         self.detect_tables()
#         self.crop_tables()


# # if __name__ == "__main__":
# #     extractor = TableExtractor()
# #     extractor.run()


from pdf2image import convert_from_path
import os
import torch
from config import Config
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image

class TableExtractor:
    def __init__(self, pdf_path=Config.DATA_PATH, output_dir="./data"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.table_output_dir = os.path.join(output_dir, "tables")
        os.makedirs(self.table_output_dir, exist_ok=True)
        self.dpi = 300
        self.confidence_threshold = 0.9
        self.padding_factor = 0.05
        self.images = []
        self.results = []
        self.table_images = []
        self.image_processor = None
        self.detection_model = None

    def convert_pdf_to_images(self):
        pages_dir = os.path.join(self.output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        self.images = convert_from_path(self.pdf_path, dpi=self.dpi)
        for i, image in enumerate(self.images):
            image_path = os.path.join(pages_dir, f"page_{i+1}.png")
            image.save(image_path, "PNG")

    def load_detection_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.detection_model.eval()

    def detect_tables(self):
        pages_dir = os.path.join(self.output_dir, "pages")
        image_paths = [os.path.join(pages_dir, f) for f in os.listdir(pages_dir) if f.startswith("page_") and f.endswith(".png")]
        image_paths.sort()
        self.images = [Image.open(path) for path in image_paths]
        self.results = []
        for i, image in enumerate(self.images):
            inputs = self.image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.detection_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            result = self.image_processor.post_process_object_detection(
                outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
            )[0]
            valid_boxes = []
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                x1, y1, x2, y2 = box.tolist()
                width, height = x2 - x1, y2 - y1
                min_width = image.width * 0.1
                min_height = image.height * 0.05
                if width >= min_width and height >= min_height:
                    valid_boxes.append((box, score, label))
            self.results.append(valid_boxes)

    def crop_tables(self):
        if not self.results:
            return
        self.table_images = []
        for i, (image, result) in enumerate(zip(self.images, self.results)):
            if not result:
                continue
            for j, (box, _, _) in enumerate(result):
                x, y, x2, y2 = box.tolist()
                padding_x = (x2 - x) * self.padding_factor
                padding_y = (y2 - y) * self.padding_factor
                x, y = max(0, x - padding_x), max(0, y - padding_y)
                x2, y2 = min(image.width, x2 + padding_x), min(image.height, y2 + padding_y)
                table_image = image.crop((x, y, x2, y2))
                table_path = os.path.join(self.table_output_dir, f"page_{i+1}_table_{j+1}.png")
                table_image.save(table_path)
                self.table_images.append(table_image)

    def run(self):
        self.convert_pdf_to_images()
        self.load_detection_model()
        self.detect_tables()
        self.crop_tables()