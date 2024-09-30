# ReceiptNinja
![Logo](https://github.com/user-attachments/assets/10b0d10b-74e4-4944-85e6-325d516b22a7)

ReceiptNinja is an intelligent web application that leverages Google Gemini and the open-source OCR model Doctr to automatically extract key information from various types of receipts, including physical, digital, images, and PDFs. With advanced capabilities, it identifies essential details such as store name, date of purchase, total amount, itemized list with prices, tax breakdowns, and payment methods. ReceiptNinja is ideal for individuals, businesses, and larger personal finance solutions, streamlining expense tracking and simplifying the management of financial data.

# Getting Started

This guide will walk you through the steps to get the project up and running on your local machine.

## 1. Set Up a Virtual Environment

We recommend using Python's built-in virtual environment tool to manage dependencies. Navigate to the project directory and follow the steps below to create and activate a virtual environment. (python version used 3.11.4, in case if it helps otherwise create with the version available at your system)


## Step-1. Install Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

To verify the installed libraries in your virtual environment, you can run:

```bash
pip list
```

## Step-2. Set Your Gemini API Key

You need to set the gemini api key as environment varible
 
```bash
export API_KEY=<YOUR_API_KEY>
```


## Step-3. Running the Program


### Example Command

```bash
python receipt_data_extractor.py --file_path "<path_to_file_or_folder>" 
```

### Argument Details:

- **`file_path`**: The path to the file or folder you want to process.



Replace `<path_to_file_or_folder>` with the actual path to the file or folder.

## 5. Output

The program generates various outputs as mentioned in the prompt:

<img width="517" alt="Screenshot 2024-09-30 at 10 14 08 PM" src="https://github.com/user-attachments/assets/bb83d007-35bc-421c-8b5c-ec3aee74e67c">

## Notes:
- The OCR models and associated files are stored in the `models/` folder, which is created automatically during execution.
- Prompt templates and other variables used in the extraction process are defined in the `variables.py` file.


