# Define image processing parameters
sharpness_factor = 3.0 # sharpness factor for image processing


max_tokens =   1300 # Max number of tokens to generate
temperature =  0  # Deterministic output from the model: keep 0
top_p=  1  # Consider the entire probability distribution: keep 1

            
            
# Define prompt to use 
prompt = '''1. **Input:** You will receive an image containing details from a shopping or food store bill. The content may vary in format and may be in different languages. Use your own internal vision capabilities to accurately extract the relevant text directly from the image, without relying on external OCR libraries like Tesseract or any other Python-based tools. Additionally, an OCR (Optical Character Recognition) output will be provided as a reference. The OCR text may contain errors or inaccuracies, so your primary task is to use your own vision capabilities to extract the correct details directly from the image.

                    2. **Objective:** Your task is to extract specific details from the bill and return them as a formatted JSON object. Use the exact key names provided below and ensure that all data is translated to English. If any detail is missing, unclear, or unreadable, follow the error handling instructions outlined below.

                    3. **Extraction Rules:**
                    - **Store Name ("store_name")**: Extract the full name of the store from which the bill originates. Ensure the name is accurate and complete.(Always present in image)
                    - **Store Address ("store_address")**: Extract the full address of the store, including street, city, postal code, and country if available. (Always present in image)
                    - **Total Amount ("total_amount")**: Extract the total amount charged on the bill. Interpret the currency based on the image and store it separately.
                    - **Currency ("currency")**: Extract the currency of the total amount, which may be in various formats such as symbols (e.g., $, €, RM) or abbreviations (e.g., USD, EUR, MYR).
                    - **Bill Date ("bill_date")**: Extract the date of the transaction. Format this as **YYYY-MM-DD**. If the time is also present, include it in the format **YYYY-MM-DD HH:MM**.
                    - **Payment Method ("payment_method")**: Extract the method of payment used (e.g., cash, credit card, debit card). If multiple methods are listed, extract every method(e.g., cash, credit card, debit card, coupon) that is used for the transaction.

                    4. **Error Handling:**
                    - If any detail cannot be extracted or is unclear, prefix the value of the relevant field with "ERROR:" and include an explanation of the issue.
                        Example:
                        ```json
                        {
                            "store_name": "ERROR: Store name not visible...(more details)",
                            "store_address": "123 Example Street, Example City, EX 12345, USA",
                            "total_amount": 92.50,
                            "currency": "USD",
                            "bill_date": "2023-09-01 14:30",
                            "payment_method": ["coupon","Credit Card"]
                        }
                        ```

                    5. **OCR Text for Reference:**
                    - Use the OCR text as a supplementary reference only. If you cannot confidently extract the information from the image alone, you may use the OCR text as a hint to guide you, but always prioritize your own extraction over the OCR data.
                    - OCR text: 
'''
