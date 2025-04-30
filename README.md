# Haystack Project

This project leverages the Haystack framework to build pipelines for document processing and question answering. To use this project, you need to set up a Python virtual environment and provide an OpenAI API key.

## Prerequisites

1. **Python**: Ensure you have Python 3.10 or higher installed on your system.
2. **OpenAI API Key**: Obtain an API key from OpenAI and set it in a `.env` file.

## Setup Instructions

### 1. Create a Virtual Environment

Run the following commands to create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 2. Install Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r notebooks/requirements.txt
```

### 3. Configure the `.env` File

Create a `.env` file in the project root directory and add your OpenAI API key:

```plaintext
OPENAI_API_KEY="your-openai-api-key"
```

Replace `your-openai-api-key` with your actual OpenAI API key.

## Project Structure

The workspace is organized as follows:

```
example.py
indexing_pipeline.yaml
pdf_rag.py
query_pipeline.yaml
wip-branching.py
data/
    riunionie-smistamenti.txt
    travel-guide.pdf
notebooks/
    haystack.ipynb
    requirements.txt
    welcome-book-rag.ipynb
    data/
        riunionie-smistamenti.txt
        travel-guide.pdf
```

### Key Files and Directories

- **`pdf_rag.py`**: Main script for running the question-answering pipeline.
- **`notebooks/`**: Contains Jupyter notebooks for exploring and testing the pipelines.
- **`data/`**: Directory for storing input documents.
- **`requirements.txt`**: Lists the dependencies required for the project.

## Dependencies

The following dependencies are required for this project:

- `python-dotenv==1.0.1`
- `haystack-ai==2.2.4`
- `haystack-experimental==0.9.0`
- `sentence-transformers==3.0.1`
- `transformers==4.42.3`
- `gradio==4.37.2`
- `huggingface_hub==0.23.4`
- `cohere-haystack==1.1.3`
- `newspaper3k==0.2.8`
- `colorama==0.4.6`
- `trafilatura==1.11.0`
- `pypdf==3.15.0`

## Usage

Once the setup is complete, you can run the script in the terminal:

```bash
python pdf_rag.py
```

You can then input your questions in the terminal, and the program will provide answers based on the processed documents. Type `exit` to quit the program.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.