# import cv2
# import numpy as np
# import pytesseract
# import pandas as pd
# import random
# import os
# from table import TableExtractor  
# import json

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# class OcrTableExtractor:
#     def __init__(self, image_dir="D:/rag-chatbot/data/tables", output_dir="D:/rag-chatbot/data/output"):
#         """Initialize the OCR Table Extractor with directories and default parameters."""
#         self.image_dir = image_dir
#         self.output_dir = output_dir
#         self.default_row_threshold = 15
#         self.default_column_threshold = 60
#         os.makedirs(self.output_dir, exist_ok=True)

#     def extract_words_bounding_boxes(self, image_path):
#         """
#         Extract words and bounding boxes from an image using Tesseract OCR.
#         """
#         gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#         custom_config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
#         data = pytesseract.image_to_data(
#             thresh,
#             output_type=pytesseract.Output.DICT,
#             config=custom_config
#         )

#         words = []
#         for i in range(len(data["text"])):
#             text = data["text"][i].strip()
#             conf = int(data["conf"][i])
#             if text and conf > 5:
#                 x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
#                 words.append((x, y, w, h, text))

#         return words, gray


#     def group_words_into_rows(self, words, row_threshold=10):
#         """Group words into rows based on vertical proximity."""
#         words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
#         rows = []
#         current_row = []
#         last_y = None

#         for (x, y, w, h, text) in words_sorted:
#             if last_y is None:
#                 current_row.append((x, y, w, h, text))
#                 last_y = y
#             else:
#                 if abs(y - last_y) <= row_threshold:
#                     current_row.append((x, y, w, h, text))
#                     last_y = y
#                 else:
#                     rows.append(current_row)
#                     current_row = [(x, y, w, h, text)]
#                     last_y = y

#         if current_row:
#             rows.append(current_row)

#         return rows

#     def split_row_into_columns(self, row, column_threshold=50):
#         """Split a row into columns based on horizontal gaps."""
#         row_sorted = sorted(row, key=lambda w: w[0])
#         columns = []
#         current_col = []
#         last_x_right = None

#         for (x, y, w, h, text) in row_sorted:
#             if last_x_right is None:
#                 current_col.append((x, y, w, h, text))
#                 last_x_right = x + w
#             else:
#                 if x - last_x_right > column_threshold:
#                     columns.append(current_col)
#                     current_col = [(x, y, w, h, text)]
#                 else:
#                     current_col.append((x, y, w, h, text))
#                 last_x_right = x + w

#         if current_col:
#             columns.append(current_col)

#         return columns

#     def build_table(self, rows, column_threshold=50):
#         """Build a 2D table from rows by splitting into columns."""
#         table = []
#         max_cols = 0

#         for row in rows:
#             col_groups = self.split_row_into_columns(row, column_threshold=column_threshold)
#             row_cells = []
#             for col in col_groups:
#                 col_text = " ".join([c[4] for c in col])
#                 row_cells.append(col_text.strip())
#             table.append(row_cells)
#             if len(row_cells) > max_cols:
#                 max_cols = len(row_cells)

#         for r in table:
#             while len(r) < max_cols:
#                 r.append("")
#         return table

#     def table_to_dataframe(self, table):
#         """Convert a 2D table into a Pandas DataFrame."""
#         df = pd.DataFrame(table)
#         return df

#     def visualize_table_structure(self, image, rows, column_threshold=50):
#         """Visualize the table structure with colored bounding boxes."""
#         if len(image.shape) == 2:
#             vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         else:
#             vis_img = image.copy()

#         for row in rows:
#             col_groups = self.split_row_into_columns(row, column_threshold=column_threshold)
#             for col in col_groups:
#                 color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
#                 for (x, y, w, h, text) in col:
#                     cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
#                     cv2.putText(vis_img, text, (x, max(y - 5, 0)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

#         return vis_img

#     def find_optimal_threshold(self, diffs):
#         """Find the optimal threshold based on the largest gap in differences."""
#         if not diffs:
#             return 0
#         diffs_sorted = sorted(diffs)
#         max_gap = 0
#         threshold = 0
#         for i in range(len(diffs_sorted) - 1):
#             gap = diffs_sorted[i + 1] - diffs_sorted[i]
#             if gap > max_gap:
#                 max_gap = gap
#                 threshold = (diffs_sorted[i] + diffs_sorted[i + 1]) / 2
#         return threshold

#     def extract_table_from_image(self, image_path, save_viz=True):
#         """Extract a table from an image with dynamic thresholds."""
#         # Step 1: Get bounding boxes
#         words, gray_image = self.extract_words_bounding_boxes(image_path)

#         # Step 2: Compute dynamic row_threshold
#         words_sorted_y = sorted(words, key=lambda w: w[1])
#         diff_y = [words_sorted_y[i + 1][1] - words_sorted_y[i][1] for i in range(len(words_sorted_y) - 1)]
#         row_threshold = self.find_optimal_threshold(diff_y)
#         if row_threshold <= 5:
#             row_threshold = self.default_row_threshold
#         # print(f"Computed row_threshold: {row_threshold}")

#         # Group into rows
#         rows = self.group_words_into_rows(words, row_threshold=row_threshold)

#         # Step 3: Compute dynamic column_threshold
#         all_gaps = []
#         for row in rows:
#             row_sorted_x = sorted(row, key=lambda w: w[0])
#             for i in range(len(row_sorted_x) - 1):
#                 x1_right = row_sorted_x[i][0] + row_sorted_x[i][2]
#                 x2_left = row_sorted_x[i + 1][0]
#                 gap = x2_left - x1_right
#                 if gap > 0:
#                     all_gaps.append(gap)
#         column_threshold = self.find_optimal_threshold(all_gaps)
#         if column_threshold <= 10:
#             column_threshold = self.default_column_threshold
#         # print(f"Computed column_threshold: {column_threshold}")

#         # Step 4: Build table
#         table = self.build_table(rows, column_threshold=column_threshold)

#         # Step 5: Convert to DataFrame
#         df = self.table_to_dataframe(table)

#         # Visualization
#         if save_viz:
#             vis_img = self.visualize_table_structure(gray_image, rows, column_threshold=column_threshold)
#             out_vis_path = os.path.join(self.output_dir, os.path.basename(image_path).replace(".png", "_viz.png"))
#             cv2.imwrite(out_vis_path, vis_img)
#             # print(f"Visualization saved to {out_vis_path}")


#         # Save JSON instead of CSV
#         out_json_path = os.path.join(self.output_dir, os.path.basename(image_path).replace(".png", "_extracted.json"))
#         with open(out_json_path, "w", encoding="utf-8") as f:
#             # Using orient="split" to preserve column labels and data
#             json.dump(df.to_dict(orient="split"), f, ensure_ascii=False, indent=2)
#         # print(f"Extracted table saved to {out_json_path}")


#         return df, vis_img
    

#     def process_all_tables(self):
#         """Process all table images extracted by TableExtractor."""
#         all_table_images = [
#             os.path.join(self.image_dir, f)
#             for f in os.listdir(self.image_dir)
#             if f.endswith(".png")
#         ]

#         # Process each table image
#         for image_path in all_table_images:
#             # Check for JSON output instead of CSV
#             json_path = os.path.join(self.output_dir, os.path.basename(image_path).replace(".png", "_extracted.json"))
#             if not os.path.exists(json_path):  # Only process if JSON doesn't exist
#                 self.extract_table_from_image(image_path)


# # if __name__ == "__main__":
# #     extractor = OcrTableExtractor()
# #     extractor.process_all_tables()

import cv2
import numpy as np
import pytesseract
import pandas as pd
import os
from table import TableExtractor
import json

class OcrTableExtractor:
    def __init__(self, image_dir="./data/tables", output_dir="./data/output"):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.default_row_threshold = 15
        self.default_column_threshold = 60
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_words_bounding_boxes(self, image_path):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        custom_config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config=custom_config)
        words = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if text and conf > 5:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                words.append((x, y, w, h, text))
        return words, gray

    def group_words_into_rows(self, words, row_threshold=10):
        words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
        rows = []
        current_row = []
        last_y = None
        for (x, y, w, h, text) in words_sorted:
            if last_y is None:
                current_row.append((x, y, w, h, text))
                last_y = y
            else:
                if abs(y - last_y) <= row_threshold:
                    current_row.append((x, y, w, h, text))
                    last_y = y
                else:
                    rows.append(current_row)
                    current_row = [(x, y, w, h, text)]
                    last_y = y
        if current_row:
            rows.append(current_row)
        return rows

    def split_row_into_columns(self, row, column_threshold=50):
        row_sorted = sorted(row, key=lambda w: w[0])
        columns = []
        current_col = []
        last_x_right = None
        for (x, y, w, h, text) in row_sorted:
            if last_x_right is None:
                current_col.append((x, y, w, h, text))
                last_x_right = x + w
            else:
                if x - last_x_right > column_threshold:
                    columns.append(current_col)
                    current_col = [(x, y, w, h, text)]
                else:
                    current_col.append((x, y, w, h, text))
                last_x_right = x + w
        if current_col:
            columns.append(current_col)
        return columns

    def build_table(self, rows, column_threshold=50):
        table = []
        max_cols = 0
        for row in rows:
            col_groups = self.split_row_into_columns(row, column_threshold=column_threshold)
            row_cells = []
            for col in col_groups:
                col_text = " ".join([c[4] for c in col])
                row_cells.append(col_text.strip())
            table.append(row_cells)
            if len(row_cells) > max_cols:
                max_cols = len(row_cells)
        for r in table:
            while len(r) < max_cols:
                r.append("")
        return table

    def table_to_dataframe(self, table):
        return pd.DataFrame(table)

    def find_optimal_threshold(self, diffs):
        if not diffs:
            return 0
        diffs_sorted = sorted(diffs)
        max_gap = 0
        threshold = 0
        for i in range(len(diffs_sorted) - 1):
            gap = diffs_sorted[i + 1] - diffs_sorted[i]
            if gap > max_gap:
                max_gap = gap
                threshold = (diffs_sorted[i] + diffs_sorted[i + 1]) / 2
        return threshold

    def extract_table_from_image(self, image_path, save_viz=False):
        words, gray_image = self.extract_words_bounding_boxes(image_path)
        words_sorted_y = sorted(words, key=lambda w: w[1])
        diff_y = [words_sorted_y[i + 1][1] - words_sorted_y[i][1] for i in range(len(words_sorted_y) - 1)]
        row_threshold = self.find_optimal_threshold(diff_y)
        if row_threshold <= 5:
            row_threshold = self.default_row_threshold
        rows = self.group_words_into_rows(words, row_threshold=row_threshold)
        all_gaps = []
        for row in rows:
            row_sorted_x = sorted(row, key=lambda w: w[0])
            for i in range(len(row_sorted_x) - 1):
                x1_right = row_sorted_x[i][0] + row_sorted_x[i][2]
                x2_left = row_sorted_x[i + 1][0]
                gap = x2_left - x1_right
                if gap > 0:
                    all_gaps.append(gap)
        column_threshold = self.find_optimal_threshold(all_gaps)
        if column_threshold <= 10:
            column_threshold = self.default_column_threshold
        table = self.build_table(rows, column_threshold=column_threshold)
        df = self.table_to_dataframe(table)
        out_json_path = os.path.join(self.output_dir, os.path.basename(image_path).replace(".png", "_extracted.json"))
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="split"), f, ensure_ascii=False, indent=2)
        return df, None

    def process_all_tables(self):
        all_table_images = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(".png")]
        for image_path in all_table_images:
            json_path = os.path.join(self.output_dir, os.path.basename(image_path).replace(".png", "_extracted.json"))
            if not os.path.exists(json_path):
                self.extract_table_from_image(image_path)