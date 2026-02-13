# Complete README.md


# 📚 PDF Question Answering with RAG

A powerful Streamlit application that enables users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG) powered by Claude AI.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Claude](https://img.shields.io/badge/Claude-AI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Overview

This application combines the power of Large Language Models (LLMs) with document retrieval to provide accurate, context-aware answers to questions about uploaded PDF documents. Using RAG (Retrieval-Augmented Generation), it first finds relevant sections of your document and then uses Claude AI to generate precise answers.

### 🎬 Demo

![Demo Screenshot](https://github.com/user-attachments/assets/79d3056f-0a20-45cb-bf11-7b15a5a74991)

*Upload any PDF → Ask questions → Get AI-powered answers*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 **PDF Upload** | Upload any PDF document for analysis |
| 🔍 **Semantic Search** | FAISS-powered vector search for relevant content |
| 🤖 **Claude AI** | Powered by Anthropic's Claude models |
| 💬 **Chat History** | Maintains conversation history within session |
| ⚙️ **Configurable** | Adjustable chunk size, overlap, and temperature |
| 📊 **Processing Stats** | View pages, chunks, and processing time |
| 📝 **Comprehensive Logging** | Detailed logs for debugging and monitoring |
| 🎨 **Clean UI** | Responsive, user-friendly Streamlit interface |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                    (Streamlit Application)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PDF PROCESSING PIPELINE                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   PyPDF     │───▶│   Text      │───▶│  OpenAI Embeddings  │  │
│  │   Loader    │    │   Splitter  │    │  (text-embedding-3) │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         VECTOR STORE                             │
│                      (FAISS In-Memory)                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Retriever  │───▶│   Prompt    │───▶│     Claude AI       │  │
│  │  (Top-K)    │    │   Template  │    │  (Response Gen)     │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain
- **LLM Provider**: Anthropic Claude
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF
- **Language**: Python 3.9+

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.9 or higher** installed
- **Anthropic API key** - [Get one here](https://console.anthropic.com/)
- **OpenAI API key** - [Get one here](https://platform.openai.com/api-keys)
- **Git** (for cloning the repository)

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-rag-app.git
cd pdf-rag-app
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Anthropic API Key
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# OpenAI API Key (for embeddings)
OPENAI_API_KEY=sk-your-api-key-here
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
pdf-rag-app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── .env.example           # Example environment file
├── .gitignore             # Git ignore file
├── README.md              # This file
├── LICENSE                # MIT License
├── logs/                  # Application logs (auto-created)
│   └── app_YYYYMMDD.log
├── assets/                # Images and assets
│   └── demo-screenshot.png
└── .streamlit/            # Streamlit configuration
    └── config.toml
```

---

## 🎮 Usage Guide

### 1. Upload a PDF

- Click the **"Browse files"** button or drag and drop a PDF
- Wait for processing to complete
- View processing statistics (pages, chunks, time)

### 2. Configure Settings (Optional)

Use the sidebar to adjust:

| Setting | Description | Default | Range |
|---------|-------------|---------|-------|
| **Model** | Claude model to use | claude-3-haiku | haiku, sonnet |
| **Temperature** | Response creativity | 0.2 | 0.0 - 1.0 |
| **Chunk Size** | Text chunk size | 1000 | 100 - 2000 |
| **Chunk Overlap** | Overlap between chunks | 200 | 0 - 500 |

### 3. Ask Questions

- Type your question in the input field
- Click **"Ask"** or press Enter
- View the AI-generated response

### 4. Review Chat History

- Scroll down to see previous Q&A pairs
- Each entry shows timestamp and model used
- Click **"Clear Chat History"** to reset

---

## ⚙️ Configuration Options

### Model Selection

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| claude-3-haiku | ⚡ Fast | Good | $ | Quick queries |
| claude-sonnet-4-5-20250929 | Medium | Excellent | $$ | Complex analysis |

### Chunk Settings Guide

| Document Type      | Recommended Chunk Size | Overlap |
|--------------------|------------------------|---------|
| Technical docs     | 1000-1500              | 200     |
| Legal documents    | 800-1000               | 150     |
| Articles/blogs     | 500-800                | 100     |
| Books              | 1500-2000              | 300     |

---

## 📊 Logging

The application includes comprehensive logging for monitoring and debugging.

### Log Location

```
logs/app_YYYYMMDD.log
```

### Log Levels

| Level | Description |
|-------|-------------|
| DEBUG | Detailed information for debugging |
| INFO | General operational information |
| WARNING | Warning messages for potential issues |
| ERROR | Error messages for failures |
| CRITICAL | Critical errors requiring immediate attention |

### Viewing Logs

**Option 1**: Click **"View Logs"** button in the app footer

**Option 2**: Access log files directly:
```bash
# View today's logs
cat logs/app_$(date +%Y%m%d).log

# Follow logs in real-time
tail -f logs/app_$(date +%Y%m%d).log
```

---

## 🌐 Deployment

### Deploy to Streamlit Cloud

#### Step 1: Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

#### Step 2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository and branch
5. Set main file path: `app.py`

#### Step 3: Configure Secrets

In Streamlit Cloud dashboard:

1. Click **"Advanced settings"**
2. Add secrets in TOML format:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
OPENAI_API_KEY = "sk-your-key-here"
```

#### Step 4: Deploy

Click **"Deploy!"** and wait for deployment to complete.

Your app will be available at:
```
https://yourusername-pdf-rag-app-xxxxx.streamlit.app
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Missing API keys" Error

```
❌ Missing API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY
```

**Solution**: 
- Ensure `.env` file exists with valid API keys
- For Streamlit Cloud, add secrets in the dashboard

#### 2. PDF Processing Fails

```
❌ Failed to process PDF
```

**Solutions**:
- Ensure PDF is not password-protected
- Check if PDF contains readable text (not scanned images)
- Try a different PDF to verify the app works

#### 3. Slow Processing

**Solutions**:
- Reduce chunk size for faster processing
- Use claude-3-haiku for quicker responses
- Consider smaller PDF files

#### 4. Out of Memory

**Solutions**:
- Reduce chunk size
- Process smaller PDFs
- Restart the application

#### 5. Empty Responses

**Solutions**:
- Ensure your question is related to the document content
- Try rephrasing your question
- Check if the PDF has extractable text

---

## 🔒 Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use Streamlit Secrets** for production deployments
3. **Rotate API keys** periodically
4. **Monitor usage** in API provider dashboards
5. **Set usage limits** in API provider settings

---

## 📈 Performance Tips

1. **Optimize chunk size** based on your document type
2. **Use appropriate overlap** to maintain context
3. **Select the right model** for your use case
4. **Cache processed documents** (handled automatically by session state)

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

### Step 1: Fork the Repository

Click the **"Fork"** button on GitHub.

### Step 2: Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Step 3: Make Your Changes

- Write clean, documented code
- Follow existing code style
- Add logging for new features

### Step 4: Test Your Changes

```bash
streamlit run app.py
# Test all functionality
```

### Step 5: Submit a Pull Request

```bash
git add .
git commit -m "Add: description of your feature"
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

### Code Style Guidelines

- Use descriptive variable names
- Add docstrings to functions
- Include logging statements
- Handle exceptions gracefully

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [LangChain](https://langchain.com/) - For the LLM orchestration framework
- [Anthropic](https://anthropic.com/) - For Claude AI
- [OpenAI](https://openai.com/) - For embeddings API
- [FAISS](https://github.com/facebookresearch/faiss) - For vector similarity search

---

## 📧 Support

For issues and questions:

- **GitHub Issues**: [Open an issue](https://github.com/yourusername/pdf-rag-app/issues)
- **Discussions**: [Start a discussion](https://github.com/yourusername/pdf-rag-app/discussions)

---

## 🗺️ Roadmap

### Upcoming Features

- [ ] Multi-PDF support
- [ ] Export chat history
- [ ] Custom prompt templates
- [ ] Support for more file types (DOCX, TXT)
- [ ] Conversation memory for follow-up questions
- [ ] User authentication
- [ ] API endpoint for programmatic access

---

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/pdf-rag-app?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/pdf-rag-app?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/pdf-rag-app)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/pdf-rag-app)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/yourusername">Your Name</a>
</p>

<p align="center">
  <a href="#-pdf-question-answering-with-rag">Back to top ⬆️</a>
</p>
```

---

## Additional Files to Create

### `.env.example`

```env
# ============================================
# API KEYS
# ============================================

# Anthropic API Key
# Get yours at: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI API Key (used for embeddings)
# Get yours at: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### `LICENSE`

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 50
enableXsrfProtection = true
enableCORS = false

[browser]
gatherUsageStats = false
```

### `.gitignore`

```gitignore
# Environment
.env
venv/
env/
.venv/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/

# Logs
logs/
*.log

# IDE
.idea/
.vscode/
*.swp
*.swo
*.sublime-*

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Vector stores
*.faiss
*.pkl

# Temp files
*.tmp
*.temp

