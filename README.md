# AI Communication Skills Scorer

**Nirmaan AI Intern Case Study Submission**

An AI-powered tool that analyzes and scores students' self-introduction transcripts based on a comprehensive rubric. The system combines rule-based methods, NLP-based semantic scoring, and data-driven weighting to provide detailed feedback on communication skills.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [How to Run Locally](#how-to-run-locally)
- [Scoring Formula & Methodology](#scoring-formula--methodology)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Sample Output](#sample-output)
- [Deployment](#deployment)
- [Design Decisions](#design-decisions)

---

## 🎯 Overview

This tool evaluates student self-introductions across 5 key criteria:
1. **Content & Structure** (40%) - Salutation, keywords, flow, semantic match
2. **Speech Rate** (10%) - Words per minute analysis
3. **Language & Grammar** (20%) - Grammar errors and vocabulary richness
4. **Clarity** (15%) - Filler word detection
5. **Engagement** (15%) - Sentiment analysis

**Input:** Transcript text (pasted or uploaded as .txt file)  
**Output:** Overall score (0-100), per-criterion scores, detailed feedback, JSON format

---

## ✨ Features

### Three Scoring Approaches (As Required)

1. **Rule-Based Methods**
   - Keyword presence detection (must-have vs good-to-have)
   - Exact pattern matching for salutations
   - Word count validation
   - Filler word detection
   - Flow/structure analysis

2. **NLP-Based Semantic Scoring**
   - Sentence embeddings using `sentence-transformers` (all-MiniLM-L6-v2)
   - Cosine similarity with ideal self-introduction patterns
   - Context-aware content understanding
   - Semantic match bonus scoring

3. **Data-Driven Rubric Weighting**
   - Weighted scoring based on provided rubric
   - Normalized final score (0-100)
   - Per-criterion breakdown with sub-scores

### Additional Features
- ✅ Interactive web UI with text input and file upload
- ✅ AI-generated constructive feedback (Groq API)
- ✅ JSON output for API integration
- ✅ Downloadable reports (JSON & Text)
- ✅ Real-time semantic similarity visualization
- ✅ Comprehensive error handling

---

## 🛠 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.31.0 |
| **NLP Model** | Sentence Transformers (all-MiniLM-L6-v2) |
| **LLM API** | Groq (Llama 3.1-8B-Instant) |
| **Grammar Check** | LanguageTool (with Java fallback) |
| **Sentiment Analysis** | VADER Sentiment |
| **Data Processing** | Pandas, NumPy, scikit-learn |
| **Language** | Python 3.8+ |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Java 11+ (optional, for advanced grammar checking)
- Internet connection (for downloading models on first run)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/communication-scorer.git
cd communication-scorer
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First run will download the sentence-transformer model (~90MB). This is automatic and only happens once.

### Step 4: Configure API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Get your free Groq API key:**
1. Visit https://console.groq.com/
2. Sign up for free account
3. Generate API key
4. Copy to `.env` file

### Step 5: (Optional) Install Java for Advanced Grammar Checking

**macOS:**
```bash
brew install openjdk@11
```

**Ubuntu/Debian:**
```bash
sudo apt-get install openjdk-11-jdk
```

**Windows:**  
Download from https://adoptium.net/

---

## 🚀 How to Run Locally

### Quick Start

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the application
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

### First-Time Setup
On first run, the system will:
1. Load the sentence transformer model (~1 minute)
2. Initialize VADER sentiment analyzer
3. Configure LanguageTool (if Java available)
4. Connect to Groq API

You'll see:
```
Loading models...
✓ Using Java at: /usr/local/bin/java
✓ LanguageTool initialized successfully
✓ All models loaded successfully!
```

### Testing the System

1. Click **"📝 Load Sample Text"** to load the provided test transcript
2. Click **"🎯 Score Transcript"**
3. View detailed results including:
   - Overall score
   - Per-criterion breakdown
   - Semantic similarity analysis
   - AI-generated feedback
   - JSON output

---

## 📊 Scoring Formula & Methodology

### Overall Architecture

```
Input Transcript
    ↓
┌─────────────────────────────────────────────┐
│  Three-Pronged Scoring Approach             │
├─────────────────────────────────────────────┤
│  1. Rule-Based (30%)                        │
│     - Keyword matching                      │
│     - Pattern detection                     │
│     - Word count validation                 │
│                                             │
│  2. NLP-Based (50%)                         │
│     - Semantic embeddings                   │
│     - Cosine similarity                     │
│     - Context understanding                 │
│                                             │
│  3. Data-Driven (20%)                       │
│     - Rubric weighting                      │
│     - Normalization                         │
│     - Aggregation                           │
└─────────────────────────────────────────────┘
    ↓
Final Score (0-100) + Detailed Feedback
```

### Detailed Scoring Formula

#### 1. Content & Structure (40% weight, max 50 points)

**Components:**
- **Salutation** (5 pts): Rule-based pattern matching
  - Excellent (5): "I am excited to introduce", "thrilled"
  - Good (4): "Good morning/afternoon/evening", "Hello everyone"
  - Normal (2): "Hi", "Hello"
  - None (0): No greeting detected

- **Keyword Presence** (30 pts): Hybrid rule-based + semantic
  - Must-have (4 pts each): name, age, school/class, family, hobbies
  - Good-to-have (2 pts each): family details, location, ambition, unique fact, strengths
  
- **Flow** (5 pts): Rule-based structural analysis
  - Checks: salutation → basics → details → closing

- **Semantic Similarity** (10 pts bonus): NLP-based
  ```python
  similarity = cosine_similarity(transcript_embedding, ideal_templates)
  score = f(average_similarity)
  ```
  Where ideal templates include:
  - "I introduce myself with name, age, and educational background"
  - "I talk about my family members and relationships"
  - "I share my hobbies, interests, and activities"
  - "I mention my goals, dreams, and aspirations"
  - "I provide unique or interesting facts about myself"

**Formula:**
```
Content_Score = Salutation + Keywords + Flow + Semantic_Bonus
```

#### 2. Speech Rate (10% weight, max 10 points)

```python
WPM = (word_count / duration_seconds) * 60

if WPM > 161:      score = 2  # Too Fast
elif 141-160:      score = 6  # Fast
elif 111-140:      score = 10 # Ideal ✓
elif 81-110:       score = 6  # Slow
else:              score = 2  # Too Slow
```

#### 3. Language & Grammar (20% weight, max 20 points)

**Grammar Check** (10 pts): LanguageTool + fallback
```python
errors_per_100_words = (error_count / word_count) * 100
grammar_ratio = 1 - min(errors_per_100_words / 10, 1)

if grammar_ratio >= 0.9: score = 10
elif >= 0.7:             score = 8
elif >= 0.5:             score = 6
elif >= 0.3:             score = 4
else:                    score = 2
```

**Vocabulary Richness** (10 pts): Type-Token Ratio (TTR)
```python
TTR = unique_words / total_words

if TTR >= 0.9: score = 10
elif >= 0.7:   score = 8
elif >= 0.5:   score = 6
elif >= 0.3:   score = 4
else:          score = 2
```

#### 4. Clarity (15% weight, max 15 points)

**Filler Word Detection** (15 pts): Rule-based
```python
filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', ...]
filler_rate = (filler_count / total_words) * 100

if rate <= 3%:     score = 15
elif <= 6%:        score = 12
elif <= 9%:        score = 9
elif <= 12%:       score = 6
else:              score = 3
```

#### 5. Engagement (15% weight, max 15 points)

**Sentiment Analysis** (15 pts): VADER (NLP-based)
```python
sentiment = vader.polarity_scores(text)
positive_score = sentiment['pos']  # 0 to 1

if positive >= 0.9: score = 15
elif >= 0.7:        score = 12
elif >= 0.5:        score = 9
elif >= 0.3:        score = 6
else:               score = 3
```

### Final Score Calculation

```python
# Calculate raw total (max 110 due to semantic bonus)
raw_total = content_structure + speech_rate + grammar + clarity + engagement

# Normalize to 0-100 scale
final_score = (raw_total / 110) * 100

# Apply data-driven weighting (already built into sub-scores)
weighted_score = (
    content_structure * 0.40 +  # 40% weight
    speech_rate * 0.10 +        # 10% weight
    grammar * 0.20 +            # 20% weight
    clarity * 0.15 +            # 15% weight
    engagement * 0.15           # 15% weight
) / sum_of_weights
```

### Key Design Decisions

1. **Why 3 approaches?**
   - **Rule-based**: Fast, deterministic, catches specific patterns
   - **NLP-based**: Understands context and meaning beyond keywords
   - **Data-driven**: Ensures alignment with rubric requirements

2. **Why sentence-transformers?**
   - Pre-trained on semantic similarity tasks
   - Efficient (384-dimensional embeddings)
   - Works offline after initial download
   - No API costs

3. **Why normalize to 100?**
   - Semantic bonus can push total beyond 100
   - Normalization ensures fair comparison
   - Maintains rubric weight proportions

---

## 📁 Project Structure

```
communication-scorer/
│
├── app.py                      # Main Streamlit application
│   ├── UI components
│   ├── Input handling (text/file)
│   ├── Results visualization
│   └── Download functionality
│
├── scorer.py                   # Core scoring engine
│   ├── CommunicationScorer class
│   ├── Rule-based methods
│   ├── NLP semantic analysis
│   ├── Data-driven weighting
│   └── AI feedback generation
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── Case study for interns.xlsx # Provided rubric (optional)
│
└── venv/                       # Virtual environment (not in git)
```

---

## 📖 Usage Guide

### Option 1: Paste Text

1. Ensure app is running (`streamlit run app.py`)
2. Select **"Paste Text"** radio button
3. Enter or paste transcript in text area
4. (Optional) Click **"Load Sample Text"** for demo
5. (Optional) Enter speech duration in sidebar
6. Click **"🎯 Score Transcript"**

### Option 2: Upload File

1. Select **"Upload Text File"** radio button
2. Click **"Browse files"**
3. Upload a `.txt` file containing transcript
4. Click **"🎯 Score Transcript"**

### Understanding Results

The output shows:
- **Overall Score**: Final weighted score (0-100)
- **Metrics**: Word count, sentences, duration, WPM
- **Grade**: Excellent (80+), Good (60-79), Fair (40-59), Needs Work (<40)
- **AI Feedback**: Personalized constructive feedback
- **Semantic Analysis**: NLP similarity scores
- **Criterion Breakdown**: Expandable sections for each criterion
- **Score Summary Table**: Quick overview
- **JSON Output**: For programmatic access
- **Downloads**: JSON and text report formats

---

## 💡 Sample Output

### Input Transcript
```
Hello everyone, myself Muskan, studying in class 8th B section from 
Christ Public School. I am 13 years old. I live with my family. There 
are 3 people in my family, me, my mother and my father...
```

### Output
```json
{
  "overall_score": 71.82,
  "words": 133,
  "sentences": 11,
  "duration_seconds": 53,
  "criteria_scores": [
    {
      "criterion": "Content & Structure",
      "total_score": 39,
      "max_score": 50,
      "weight": 40,
      "subcriteria": [
        {"name": "Salutation", "score": 4, "feedback": "Good salutation"},
        {"name": "Keyword Presence", "score": 24, "feedback": "Found: name, age..."},
        {"name": "Flow", "score": 5, "feedback": "Good flow maintained"},
        {"name": "Semantic Similarity", "score": 6, "feedback": "avg: 0.427"}
      ]
    },
    ...
  ],
  "semantic_analysis": {
    "avg_similarity": 0.427,
    "max_similarity": 0.543
  }
}
```

---

## 🌐 Deployment

### Local Deployment (Recommended for Testing)

```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

### Streamlit Cloud Deployment (Free)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Visit https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Select `app.py` as main file
   - Add secrets: `GROQ_API_KEY=your_key`
   - Click "Deploy"

3. **Your app will be live at:**
   `https://[your-app-name].streamlit.app`

### AWS/Other Cloud Platforms

For AWS EC2, Azure, or GCP deployment:
1. Set up Python 3.8+ environment
2. Install dependencies
3. Configure environment variables
4. Run: `streamlit run app.py --server.port 8501`
5. Configure firewall/security groups

---

## 🧠 Design Decisions & Product Thinking

### Why This Approach?

**1. User Experience First**
- Single-page application for simplicity
- Instant feedback with loading indicators
- Clear visual hierarchy
- Downloadable results for record-keeping

**2. Scalability**
- Modular code structure (scorer.py separate from UI)
- Caching for model loading (avoids reload)
- JSON output for API integration
- Easy to add new criteria

**3. Robustness**
- Graceful fallbacks (Java optional for grammar)
- Error handling at each stage
- Works offline after initial setup
- Cross-platform compatibility

**4. Transparency**
- Shows exact scores and reasoning
- Displays semantic similarity numbers
- Explains what keywords were found/missing
- AI feedback is actionable

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Java dependency for LanguageTool | Implemented fallback basic grammar check |
| Model download time | Added caching + loading indicator |
| Varying transcript lengths | Auto-estimate duration if not provided |
| Semantic scoring threshold | Calibrated using provided sample |
| API costs | Used free Groq tier, fallback feedback |

### Future Enhancements

- [ ] Audio file upload with transcription
- [ ] Multi-language support
- [ ] Historical score tracking
- [ ] Batch processing
- [ ] Custom rubric editor
- [ ] Real-time scoring as user types
- [ ] Voice recording integration
- [ ] PDF report generation

---

## 🔧 Troubleshooting

### Common Issues

**1. "Module not found" errors**
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**2. "GROQ_API_KEY not found"**
```bash
# Solution: Create .env file with API key
echo "GROQ_API_KEY=your_key_here" > .env
```

**3. "Java not found" warning**
```bash
# Solution: Install Java or use basic grammar checking
brew install openjdk@11  # macOS
# OR just continue - app will use fallback
```

**4. Models downloading slowly**
```bash
# First run downloads ~90MB model
# Be patient, it's one-time only
# Subsequent runs are instant
```

---

## 👨‍💻 Author

**[Your Name]**  
Nirmaan AI Intern Case Study Submission  
[Your Email] | [LinkedIn] | [GitHub]

---

## 📄 License

MIT License - feel free to use and modify

---

## 🙏 Acknowledgments

- Nirmaan AI team for the opportunity
- Groq for free API access
- HuggingFace for sentence-transformers
- Open-source community

---

**Built with ❤️ for Nirmaan AI Communication Program**