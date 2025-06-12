# PDF to MCQ Generator

A Python-based application that automatically extracts text from PDF documents, identifies key concepts, and generates multiple-choice questions (MCQs) for educational purposes. The application uses AI-powered keyword extraction and question generation to create meaningful assessments from any PDF content.

## 🌐 Live Demo

**Try the hosted version**: [https://testme-4fg7.onrender.com](https://testme-4fg7.onrender.com)

## 📋 Features

- **PDF Text Extraction**: Structured text extraction from PDF documents
- **Intelligent Keyword Extraction**: AI-powered identification of key concepts and terms
- **Automatic MCQ Generation**: Creates multiple-choice questions based on extracted keywords
- **Flexible Configuration**: Customizable number of questions and keyword filtering
- **JSON Output**: Structured data export for easy integration
- **Web Interface**: Flask-based web application for easy interaction

## 🔧 How It Works

The application follows a systematic pipeline:

1. **PDF Text Extraction** (`pdfextractor.py`)
   - Extracts structured text content from uploaded PDF files
   - Saves extraction results as JSON with timestamps

2. **Keyword Extraction** (`keywords_extractor.py`)
   - Analyzes extracted text using AI to identify important keywords and concepts
   - Filters keywords based on relevance scores (default threshold: 0.35)
   - Generates up to 5 top keywords per section

3. **Question Generation** (`question_generator.py`)
   - Uses DeepSeek AI model (via OpenRouter API) to create MCQs
   - Generates contextually relevant multiple-choice questions
   - Configurable number of questions (default: 10)

4. **Output Management**
   - Saves all intermediate results (text extraction, keywords, MCQs) as timestamped JSON files
   - Organizes outputs in the `data/` directory

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7+
- OpenRouter API key for DeepSeek model access

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd pdf-mcq-generator
```

### Step 2: Install Dependencies

Navigate to the backend folder and install required packages:

```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Configure API Key

Set up your OpenRouter API key for DeepSeek model access:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or create a `.env` file in the backend directory:

```
OPENROUTER_API_KEY=your-api-key-here
```

### Step 4: Run the Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
pdf-mcq-generator/
├── backend/
│   ├── app.py                 # Flask web application
│   ├── overall_flow.py        # Main processing pipeline
│   ├── pdfextractor.py        # PDF text extraction module
│   ├── keywords_extractor.py  # AI-powered keyword extraction
│   ├── question_generator.py  # MCQ generation using DeepSeek
│   ├── jsonfilter.py         # Keyword filtering utilities
│   ├── requirements.txt       # Python dependencies
│   └── data/                  # Output directory for generated files
└── README.md
```

## 🔄 Processing Pipeline

1. **Upload PDF** → Text extraction with structured formatting
2. **Extract Keywords** → AI identifies key concepts (top 5 per section)
3. **Filter Keywords** → Relevance-based filtering (threshold: 0.35)
4. **Generate MCQs** → AI creates multiple-choice questions (default: 10)
5. **Export Results** → JSON files with timestamps for all stages

## ⚙️ Configuration Options

You can customize the processing by modifying parameters in `overall_flow.py`:

- **Number of keywords**: Change the `5` parameter in `extract_keywords_from_pdf_json()`
- **Keyword threshold**: Modify the `0.35` value in `filter_keywords_from_file()`
- **Number of MCQs**: Adjust the `10` parameter in `generate_mcqs()`

## 🤖 AI Integration

The application leverages:
- **DeepSeek Model**: Advanced AI for keyword extraction and question generation
- **OpenRouter API**: Reliable API gateway for AI model access
- **Structured Prompting**: Optimized prompts for educational content generation

## 📊 Output Format

All outputs are saved as JSON files with timestamps:

- `text_extraction_result_[filename]_[timestamp].json`
- `keyword_extraction_result_[filename]_[timestamp].json`
- `mcqs_[filename]_[timestamp].json`

## 🛠️ Troubleshooting

**Common Issues:**

1. **API Key Errors**: Ensure your OpenRouter API key is properly configured
2. **PDF Processing Failures**: Check if the PDF is text-based (not scanned images)
3. **Dependency Issues**: Make sure all requirements are installed: `pip install -r requirements.txt`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📝 License

This project is open source. Please check the license file for details.

## 🔗 Links

- **Hosted Application**: [https://testme-4fg7.onrender.com](https://testme-4fg7.onrender.com)
- **OpenRouter**: [https://openrouter.ai](https://openrouter.ai)
- **DeepSeek**: AI model used for content generation
