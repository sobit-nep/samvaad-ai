# संवाद AI: Intelligent Docs Interaction 💬📚

## Overview
**संवाद AI** is a Streamlit-based application designed for intelligent interaction with PDF documents. This project leverages advanced language models and Google Generative AI to provide detailed answers to user queries in both English and Nepali. Key features include:

- Parsing and processing PDF documents.
- Providing context-aware answers using FAISS for vector-based search.
- Supporting multilingual queries with transliteration for Romanized Nepali text.

## Features
- **PDF Document Processing**: Extract text from PDF files and store it in a FAISS vector store for similarity search.
- **Multilingual Question Answering**: Supports both English and Nepali queries, with seamless transliteration for Romanized Nepali.
- **Customizable Prompts**: Tailored prompts for Nepali and English provide accurate, context-driven responses.

## Installation
Follow these steps to set up and run the project locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sobit-nep/samvaad-ai.git
   cd samvaad-ai
   ```

2. **Set Up a Virtual Environment**
   Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root and add your Google API key (create here: https://aistudio.google.com/apikey):
   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

5. **Run the Application**
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Upload PDFs**:
   - Use the sidebar to upload your PDF documents.
   - Click the "Process" button to extract text and build a vector store.

2. **Ask Questions**:
   - Enter your question in the input field on the main page.
   - Receive detailed, context-based answers in either English or Nepali.

## Project Structure
- `app.py`: Main application code.
- `requirements.txt`: List of required Python libraries.
- `.env`: File for environment variables (Google API key).
- `faiss_index/`: Directory for the FAISS vector index.

## Requirements
- Python 3.8 or higher
- Google Generative AI API access
- Internet connection for API calls

## Key Technologies
- **Streamlit**: For the user interface.
- **PyPDF2**: For extracting text from PDF files.
- **FAISS**: For vector-based similarity search.
- **Langchain**: For prompt templates and chain orchestration.
- **Google Generative AI**: For embeddings and language models.
- **Langdetect**: For language detection.

## Future Enhancements
- Add support for additional languages.
- Optimize text processing for large documents.
- Explore integrations with other AI models for improved performance.

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests to enhance the project.

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.
