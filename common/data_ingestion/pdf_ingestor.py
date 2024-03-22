import concurrent.futures
import os
from typing import List

import cv2
import numpy as np
import pytesseract
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader

from .text_ingestor import TextIngestor


class PDFIngestor(TextIngestor):
    def __init__(self):
        super().__init__()

    def extract_text(self, pdf_path: str, categories: List[str]) -> List[Document]:
        """
        Extracts text from PDF file at the given path.
        """

        print("Extracting text from PDF...")

        # Check if the PDF contains text
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            text = reader.pages[0].extract_text()

        # If the PDF contains text, load it normally
        if text:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

        # If the PDF does not contain text, assume the pages are images and process them with OCR
        else:
            documents = self.ocr_process_pdf(pdf_path)

        print("Splitting documents...")
        chunks = self.text_splitter.split_documents(documents)

        # Populate the metadata
        source = os.path.basename(pdf_path)
        for i, doc in enumerate(chunks):
            doc.page_content = doc.page_content.strip()
            doc.metadata["source"] = source
            doc.metadata["categories"] = categories
            doc.metadata["doc_index"] = i

        return chunks

    def ocr_process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process a PDF with OCR.
        """
        print("Converting PDF to images...")
        # Convert the PDF to images
        images = convert_from_path(pdf_path)

        print("Processing images with OCR...")
        # Create a list to hold the future results.
        future_to_image = {}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i, image in enumerate(images):
                # If the image appears to contain two book pages, split it
                if image.width > image.height:
                    left = image.crop((0, 0, image.width // 2, image.height))
                    right = image.crop((image.width // 2, 0, image.width, image.height))
                    images[i : i + 1] = [left, right]

                print(f"Dewarping and OCRing image {i+1}/{len(images)}...")
                # Dewarp the page and save it as a temporary image
                dewarped_image_path = f"/tmp/dewarped_page_{i}.jpg"  # include the image index in the file name
                dewarped_image = self.dewarp_image(image)

                # OCR the dewarped image
                future_to_image[
                    executor.submit(pytesseract.image_to_string, dewarped_image)
                ] = i
                
        print("Generating documents...")
        documents = [None] * len(future_to_image)  # preallocate list for results

        for future in concurrent.futures.as_completed(future_to_image):
            i = future_to_image[future]
            try:
                text = future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
            else:
                # Create a Document
                document = Document(page_content=text)
                documents[i] = document

        return documents

    def dewarp_image(self, image) -> Image:
        """
        Dewarp an image and return it.
        """
        # Convert the PIL Image to an OpenCV image (numpy array)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Use the dewarp code snippet provided
        dewarped_image = self.dewarp(image_cv)

        # Convert back the dewarped image from OpenCV image (numpy array) to PIL Image
        dewarped_image_pil = Image.fromarray(
            cv2.cvtColor(dewarped_image, cv2.COLOR_BGR2RGB)
        )

        return dewarped_image_pil

    def dewarp(self, image: np.ndarray) -> np.ndarray:
        """
        Dewarps an image and returns it.
        """

        # Convert image to grayscale if it is not already
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Select largest contour
        contour = max(contours, key=cv2.contourArea)

        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Sort points in approx contour
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Compute the width and height of new image
        width_1 = np.sqrt(
            ((rect[0][0] - rect[1][0]) ** 2) + ((rect[0][1] - rect[1][1]) ** 2)
        )
        width_2 = np.sqrt(
            ((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2)
        )
        height_1 = np.sqrt(
            ((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2)
        )
        height_2 = np.sqrt(
            ((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2)
        )
        maxWidth = max(int(width_1), int(width_2))
        maxHeight = max(int(height_1), int(height_2))

        # Set destination points for perspective transformation
        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        # Compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped
