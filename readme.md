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

The program generates various outputs, which are saved in the following directories:

## Notes:
- The OCR models and associated files are stored in the `models/` folder, which is created automatically during execution.
- Prompt templates and other variables used in the extraction process are defined in the `variables.py` file.


