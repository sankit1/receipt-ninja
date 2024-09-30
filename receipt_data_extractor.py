# Importing Libraries -----------------------------------------------------------------------------------------------------------------------------------------

# Standard library imports
import os  # To work with file paths and environment variables
import io  # For in-memory byte streams
import base64  # To encode/decode images to/from base64 format
import json  # For working with JSON data
import argparse  # For parsing command-line arguments
import re  # For regular expressions, used in JSON sanitization
import tempfile  # To create temporary files

import sys  # For interacting with the system and handling command-line output
import importlib # To dynamically import modules


from PIL import Image, ImageEnhance  # For image processing and enhancement
import fitz  # PyMuPDF, for reading and processing PDF files
from doctr.models import ocr_predictor  # For OCR processing with the Doctr library
from doctr.io import DocumentFile  # For handling image documents with Doctr
from dotenv import load_dotenv  # For loading environment variables from a .env file

# Custom variables import
from variables import prompt, sharpness_factor, log_file_name, override_value, max_tokens, temperature, top_p
from variables import processed_files_folder, unprocessed_files_folder, process_type_choices, model_choices, output_ocr_text

import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])



# Importing API key -----------------------------------------------------------------------------------------------------------------------------------------

try:
    # Get the current working directory
    current_directory = os.getcwd()
  
    # Set current working directory as directory where Doctr will save models
    os.environ['DOCTR_CACHE_DIR'] = current_directory
  
except Exception as e:
    raise


# Class to extract vehicle data from images  ------------------------------------------------------------------------------------------------------------------

class ImageProcessor:
    def __init__(self, file_path, llm_model):
        """
        Initialize with the file path.
        :param file_path: Path to the file or folder to be processed
        """
        try:
            self.file_path = file_path
            self.LLM_model = llm_model
        except Exception as e:
            raise

    def is_image(self, file_path):
        """
        Check if the file is an image based on its extension.
        :param file_path: Path to the file to check.
        :return: True if the file is an image, False otherwise.
        """
        try:
            
            # Valid image extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            
            # Get the file extension and check if it matches one of the valid image extensions
            file_extension = os.path.splitext(file_path)[1].lower()
            result = file_extension in image_extensions
            
            return result
        except Exception as e:
            raise


    def rotate_image(self, image, degrees_to_rotate):
        """
        Rotate the image by a given degree with high-quality resampling to avoid pixelation.
        :param image: The image to be rotated.
        :param degrees_to_rotate: The number of degrees to rotate the image.
        :return: The rotated image.
        """
        try:
            rotated_image = image.rotate(degrees_to_rotate, resample=Image.BICUBIC, expand=True)
            return rotated_image
        except Exception as e:
            raise


    def encode_image_to_base64(self, image):
        """
        Convert an image to base64 format with MIME type.
        :param image: The image object to be encoded.
        :return: The base64-encoded string with MIME type.
        """
        try:
            
            # Save image to a BytesIO buffer in PNG format
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            
            # Encode the buffer content to base64 and decode to UTF-8 string
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return f"data:image/png;base64,{base64_image}"
        
        except Exception as e:
            raise


    def convert_pdf_to_images(self, file_path, dpi=300):
        """
        Convert PDF pages to images with a specified resolution using PyMuPDF.
        :param file_path: Path to the PDF file to be converted.
        :param dpi: The DPI resolution for converting the PDF pages to images.
        :return: A list of images (one for each page).
        """
        try:          
            # Open the PDF document
            pdf_document = fitz.open(file_path)
            images = []

            # Process each page in the PDF
            for page_number in range(pdf_document.page_count):
                try:
                    
                    # Load the page and convert to an image
                    page = pdf_document.load_page(page_number)
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Append the image to the list
                    images.append(img)
                    
                except Exception as page_error:
                    raise

            return images
        
        except Exception as e:
            raise


    def merge_images_vertically(self, images):
        """
        Merge a list of images vertically into a single image.
        :param images: A list of images to be merged vertically.
        :return: A single merged image.
        """
        try:
            
            # Calculate total height and max width for the merged image
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            
            # Create a new image with the appropriate size
            merged_image = Image.new("RGB", (max_width, total_height))
            
            # Paste each image into the new image at the correct y_offset
            y_offset = 0
            for index, img in enumerate(images):
                merged_image.paste(img, (0, y_offset))
                y_offset += img.height

            return merged_image
        
        except Exception as e:
            raise

    def enhance_image(self, image):
        """
        Enhance the sharpness and convert the image to black and white.
        :param image: The image to be enhanced.
        :return: The enhanced image.
        """
        try:
            
            # Convert the image to black and white
            img_bw = image.convert('L')
            
            # Enhance sharpness with a sharpness factor
            enhancer = ImageEnhance.Sharpness(img_bw)
            enhanced_image = enhancer.enhance(sharpness_factor)
            return enhanced_image
        
        except Exception as e:
            raise


    def process_image(self, image_path):
        """
        Process a single image, correct its orientation, enhance it, and return the result along with extracted OCR text.
        :param image_path: The file path of the image or the image object.
        :return: The enhanced image and the extracted OCR text.
        """
        try:
            
            # Load the image if an image object is passed, otherwise load from file path
            if isinstance(image_path, str):
                img_original = Image.open(image_path)
            else:
                img_original = image_path

            # Temporarily save the in-memory image to a temporary file if necessary
            if not isinstance(image_path, str):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
                    img_original.save(temp_image_file.name, format='PNG')
                    temp_image_path = temp_image_file.name
            else:
                temp_image_path = image_path

            # Load and process the image using Doctr
            model = ocr_predictor('db_resnet50', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True, detect_orientation=True)
            
            doc = DocumentFile.from_images([temp_image_path])
            result = model(doc)

            # Extract OCR text from the result
            ocr_text = " ".join([word.value for page in result.pages for block in page.blocks for line in block.lines for word in line.words])
            
            # Extract the orientation value (in degrees) from the result
            orientation = int(result.pages[0].orientation['value'])
            
            # Rotate the image if necessary
            if orientation != 0:
                img_rotated = self.rotate_image(img_original, orientation)
                img_original = img_rotated  # Update the original image for further processing
            
            # Enhance the image
            image_enhanced = self.enhance_image(img_original)

            # Clean up the temporary file if it was created
            if not isinstance(image_path, str):
                os.remove(temp_image_path)
            
            # Return the enhanced image and the extracted OCR text
            return image_enhanced, ocr_text

        except Exception as e:
            raise


    def process_pdf(self, file_path):
        """
        Process a PDF file, correct the orientation of each page, merge the images, 
        and return the final image along with the merged OCR text.
        :param file_path: The path to the PDF file.
        :return: The enhanced merged image and the combined OCR text.
        """
        try:
            
            # Convert PDF to list of images
            images = self.convert_pdf_to_images(file_path)
            processed_images = []
            merged_ocr_text = ""  # Initialize a variable to hold the combined OCR text
            
            # Process each page's image
            for page_number, img in enumerate(images):
                try:
                    
                    # Process the image (correct orientation, enhance it, etc.)
                    final_image, ocr_text = self.process_image(img)
                    
                    # Add processed image to the list and OCR text to the merged string
                    processed_images.append(final_image)
                    merged_ocr_text += ocr_text + "\n"  # Add a newline between pages for clarity
                    
                    
                except Exception as page_error:
                    raise
            
            # Merge all processed images vertically
            merged_image = self.merge_images_vertically(processed_images)
            
            # Enhance the merged image
            final_enhanced_image = self.enhance_image(merged_image)
            
            # Return the final enhanced image and the merged OCR text
            return final_enhanced_image, merged_ocr_text
        
        except Exception as e:
            raise



    def extract_information_from_image_gemini(self, image_url, ocr_text):
    
        """
        Extract vehicle-related information from an image using the OpenAI API and OCR text.
        :param image_url: The URL of the image.
        :param ocr_text: The OCR-extracted text to be used as part of the prompt.
        :return: A dictionary with the extracted vehicle information in JSON format.
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            combined_prompt = prompt + "-->" + ocr_text + "<--"
            
            generated_response = model.generate_content(combined_prompt)
            print(generated_response.text)
            print("That was gemini response")
            

            return generated_response.text

        except Exception as e:
            raise



  


    def data_extractor(self, file_path, ocr_text, filename, LLM_model):
        """
        Extract vehicle-related data from the image using OCR text and a language model.
        :param file_path: Path to the image file.
        :param ocr_text: The OCR-extracted text from the image.
        :param filename: The name of the file being processed.
        :param LLM_model: The language model to use (currently only 'openai' is supported).
        :return: A dictionary with extracted data.
        """
        try:
            if not file_path:
                return None
            
            
            # Encode the image to base64 format
            image_url = self.encode_image_to_base64(file_path)
            
            # Extract information from the image using the selected LLM model
            if LLM_model == "gemini":
                return  self.extract_information_from_image_gemini(image_url, ocr_text)

            # Extract specific fields from the vehicle information
            document = filename.split("\\")[-1].split(".")[0]
            store_name = bill_info.get('store_name', 'ERROR: Model returned invalid data format probably due to image(content,quality or size) issues.')
            store_address = bill_info.get('store_address', 'ERROR: Model returned invalid data format probably due to image(content,quality or size) issues.')
            total_amount = bill_info.get('total_amount', 'ERROR: Model returned invalid data format probably due to image(content,quality or size) issues.')
            currency = bill_info.get('currency', 'ERROR: Model returned invalid data format probably due to image(content,quality or size) issues.')
            bill_date = bill_info.get('bill_date', 'ERROR: Model returned invalid data format probably due to image(content,quality or size) issues.')
            payment_method = bill_info.get('payment_method', 'ERROR: Model returned invalid data format probably due to image(content,quality or size) issues.')
  
            
            # Create a dictionary with the extracted information
            extracted_data = {
                "document": document,
                "store_name": store_name,
                "store_address": store_address,
                "total_amount": total_amount,
                "currency": currency,
                "bill_date": bill_date,
                "payment_method": payment_method
            }
            print(extracted_data)
            return extracted_data
        
        except Exception as e:
            raise
    

    def save_ocr_text(self, ocr_text, file_name):
        """Save the OCR text to a .txt file in the output_ocr_text directory."""
        try:
            # Ensure the output directory exists
            os.makedirs(output_ocr_text, exist_ok=True)

            # Generate the file name with .txt extension
            file_name = os.path.splitext(os.path.basename(file_name))[0] + '.txt'
            output_path = os.path.join(output_ocr_text, file_name)
            
            # Save the OCR text to the file
            with open(output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(ocr_text)
            
        except OSError as os_error:
            raise

        except Exception as e:
            raise

        except Exception as e:
            raise

                         
                               
    def process_unit_file(self):
        """
        Process a single file (image or PDF) and extract data using the specified LLM model.
        :return: None
        """
        try:
            if self.is_image(self.file_path):
                try:
                    with Image.open(self.file_path) as img:
                        final_image, ocr_text = self.process_image(img)
                        self.save_ocr_text(ocr_text, self.file_path)
                        extracted_data = self.data_extractor(final_image, ocr_text, self.file_path, self.LLM_model)
                except Exception as e:
                    raise

            elif self.file_path.lower().endswith('.pdf'):
                try:
                    final_image, ocr_text = self.process_pdf(self.file_path)
                    self.save_ocr_text(ocr_text, self.file_path)
                    extracted_data = self.data_extractor(final_image, ocr_text, self.file_path, self.LLM_model)
                except Exception as e:
                    raise

            else:
                return None
            
            
        except Exception as e:
            raise



# Running the code -----------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        
        # Create the argument parser
        parser = argparse.ArgumentParser(description="Processes images or PDFs to extract vehicle-related data")
        
        # Define the required arguments
        parser.add_argument('--file_path', type=str, required=True, help="The path to the file or folder to be processed (image or PDF).")
        parser.add_argument('--llm_model', type=str, default='gemini', choices= model_choices, help="The language model to use (currently only 'openai' is supported).")
        
        # Parse the command-line arguments
        args = parser.parse_args()
        
        # Ensure the file path exists before proceeding
        if not os.path.exists(args.file_path):
            raise FileNotFoundError(f"The specified file path does not exist: {args.file_path}")
        
        # Create an instance of the ImageProcessor with the provided arguments
        processor = ImageProcessor(file_path=args.file_path, llm_model=args.llm_model)
        
        # Run the process
        processor.process_unit_file()
        
    except FileNotFoundError as fnf_error:
        raise
    
    except Exception as e:
        raise




