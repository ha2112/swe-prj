import re
import markdown
from bs4 import BeautifulSoup
import os
import PyPDF2
import tiktoken
from langchain.docstore.document import Document
import pylcs
import pandas as pd
import textwrap
from vertexai.preview import tokenization


def clean_presentation_text(text):
    """Cleans text extracted from presentations."""

    # Remove presentation headers/footers (Vietnamese and English)
    text = re.sub(r"TRƯỜNG CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG", "", text)
    text = re.sub(r"School of Information and Communication Technology", "", text)

    # Remove dates like '3/12/2025'
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{4}", "", text)

    # Remove page/slide numbers (often appear alone or with context like 'Link Layer: 6-XX')
    text = re.sub(r"Link Layer: \d+-\d+", "", text) # Remove specific "Link Layer: X-Y" references
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE) # Remove lines containing only numbers (slide numbers)
    text = re.sub(r"\n\d+\n", "\n", text) # Remove slide numbers appearing between lines
    text = re.sub(r"\s+\d+\s*$", "", text, flags=re.MULTILINE) # Remove trailing numbers on lines

    # Replace common unicode symbols used for bullets/arrows with Markdown list item
    text = re.sub(r"[\uf0a7\uf06c\uf0e8\uf0e0\uf0df❖]", "*", text) # Replace with standard Markdown bullet

    # Normalize bullet points preceded by '*' or '•'
    text = re.sub(r"\n[•*]\s*", "\n* ", text) 

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text) # Replace 3 or more newlines with just two

    # Remove leading/trailing whitespace from each line
    text = "\n".join([line.strip() for line in text.splitlines()])

    # Remove initial quote if present
    if text.startswith("'"):
        text = text[1:]
    if text.endswith("'"):
        text = text[:-1]

    # Remove lines that became empty after cleaning
    text = "\n".join([line for line in text.splitlines() if line.strip()])

    return text

def pdf_text_extract(file_path): #changed function name here
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            documents = pdf_reader.pages  # Get all pages from the PDF

            # Concatenate text from all pages
            text = " ".join([doc.extract_text() for doc in documents])
        return text
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"An error occur: {e}"

def md_text_extract(file_path):
    """
    Extracts plain text from a Markdown file.

    Args:
        file_path (str): The path to the Markdown file.

    Returns:
        str: The extracted plain text content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Convert Markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # Use BeautifulSoup to extract text from HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()

        return text.strip()

    except FileNotFoundError:
        return f"Error: File not found at '{file_path}'"
    except Exception as e:
        return f"An error occurred: {e}"

def txt_text_extract(file_path):
    """
    Extracts text from a .txt file.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        str: The text content of the file, or None if the file does not exist or an error occurs.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        # Open the file in read mode ('r')
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the entire content of the file into a string
            text = file.read()
            return text
    except Exception as e:
        # Handle any potential errors during file processing
        print(f"An error occurred while reading the file: {e}")
        return None

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing unnecessary characters, extra spaces,
    and standardizing formatting.  Handles common issues in OCR'd text.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""

    # 1. Handle Unicode issues (remove non-ASCII, normalize) - More robust
    text = text.encode('ascii', 'ignore').decode('ascii')

    # 2. Remove/replace control characters - handle newlines, tabs explicitly
    text = text.replace('\t', ' ')  # Tabs to spaces
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # Remove other control chars

    # 3.  Handle common OCR errors and formatting issues
    text = re.sub(r'(\w)-(\w)', r'\1\2', text)  # Remove hyphens between words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Split concatenated words
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Split number-letter
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Split letter-number
    text = re.sub(r' +', ' ', text)  # Remove extra spaces
    text = text.strip() # Remove leading and trailing spaces

    # 4. Fix common OCR errors (more comprehensive)
    text = re.sub(r'ﬁ', 'fi', text)
    text = re.sub(r'ﬀ', 'ff', text)
    text = re.sub(r'ﬂ', 'fl', text)
    text = re.sub(r'ﬃ', 'ffi', text)
    text = re.sub(r'ﬄ', 'ffl', text)
    text = re.sub(r'°', ' ', text)

    # 5. Handle line breaks and whitespace
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
    text = re.sub(r' +', ' ', text)  # Squeeze multiple spaces.

    return text.strip()

def count_tokens_for_gemini(str) -> int:
    model_name = "gemini-1.5-pro-002"
    tokenizer = tokenization.get_tokenizer_for_model(model_name)
    token_count = tokenizer.count_tokens(str)
    return token_count
