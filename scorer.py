import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from groq import Groq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import warnings

# Suppress torch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class CommunicationScorer:
    def __init__(self, groq_api_key):
        """Initialize the scorer with models"""
        from groq import Groq
        import shutil
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Load sentence transformer model for semantic similarity
        print("Loading models...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize VADER for sentiment analysis
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Try to initialize LanguageTool with cross-platform Java detection
        self.grammar_tool = None
        try:
            # Try to find Java automatically
            java_path = shutil.which('java')
            
            if java_path:
                java_home = os.path.dirname(os.path.dirname(java_path))
                os.environ['JAVA_HOME'] = java_home
                print(f"✓ Found Java at: {java_path}")
                
                # Try to initialize LanguageTool
                import language_tool_python
                self.grammar_tool = language_tool_python.LanguageTool('en-US')
                print("✓ LanguageTool initialized successfully")
            else:
                print("⚠️ Java not found. Using basic grammar checking instead")
                
        except Exception as e:
            print(f"⚠️ Could not initialize LanguageTool: {str(e)}")
            print("✓ Using basic grammar checking instead")
            self.grammar_tool = None
        
        print("✓ All models loaded successfully!")
        
        # Define filler words
        self.filler_words = [
            'um', 'uh', 'like', 'you know', 'so', 'actually', 
            'basically', 'right', 'i mean', 'well', 'kinda', 
            'sort of', 'okay', 'hmm', 'ah'
        ]
    
    def count_words(self, text):
        """Count words in transcript"""
        return len(text.split())
    
    def count_sentences(self, text):
        """Count sentences in transcript"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    # ===== SEMANTIC SIMILARITY SCORING =====
    
    def score_semantic_similarity(self, text):
        """Score semantic similarity to ideal self-introduction (0-10 bonus points)"""
        # Define ideal self-introduction characteristics
        ideal_templates = [
            "I introduce myself with my name, age, and educational background",
            "I talk about my family members and relationships",
            "I share my hobbies, interests, and activities I enjoy",
            "I mention my goals, dreams, and aspirations for the future",
            "I provide unique or interesting facts about myself"
        ]
        
        # Get embeddings
        transcript_embedding = self.semantic_model.encode(text, convert_to_tensor=False)
        template_embeddings = self.semantic_model.encode(ideal_templates, convert_to_tensor=False)
        
        # Calculate cosine similarity with each template
        similarities = cosine_similarity([transcript_embedding], template_embeddings)[0]
        
        # Average similarity score - Convert to Python float
        avg_similarity = float(np.mean(similarities))
        max_similarity = float(np.max(similarities))
        
        # Convert to points (0-10 scale)
        # High semantic match = more points
        if avg_similarity >= 0.7:
            score = 10
            feedback = "Excellent semantic match"
        elif avg_similarity >= 0.6:
            score = 8
            feedback = "Good semantic alignment"
        elif avg_similarity >= 0.5:
            score = 6
            feedback = "Moderate semantic match"
        elif avg_similarity >= 0.4:
            score = 4
            feedback = "Fair semantic alignment"
        else:
            score = 2
            feedback = "Weak semantic match"
        
        detailed_feedback = f"{feedback} (avg: {avg_similarity:.3f}, max: {max_similarity:.3f})"
        
        return score, detailed_feedback, avg_similarity, max_similarity
    
    # ===== CONTENT & STRUCTURE SCORING =====
    
    def score_salutation(self, text):
        """Score salutation level (0-5 points)"""
        text_lower = text.lower().strip()
        first_sentence = text_lower.split('.')[0]
        
        # Excellent (5 points)
        excellent_phrases = ['i am excited to introduce', 'feeling great', 'thrilled', 'delighted']
        for phrase in excellent_phrases:
            if phrase in first_sentence:
                return 5, "Excellent salutation"
        
        # Good (4 points)
        good_greetings = ['good morning', 'good afternoon', 'good evening', 'good day', 'hello everyone']
        for greeting in good_greetings:
            if greeting in first_sentence:
                return 4, "Good salutation"
        
        # Normal (2 points)
        normal_greetings = ['hi', 'hello']
        for greeting in normal_greetings:
            if greeting in first_sentence:
                return 2, "Normal salutation"
        
        # No salutation (0 points)
        return 0, "No salutation found"
    
    def score_keyword_presence(self, text):
        """Score keyword presence (0-30 points)"""
        text_lower = text.lower()
        score = 0
        found_keywords = []
        missing_keywords = []
        
        # Must Have keywords (4 points each, max 20)
        must_have = {
            'name': ['name', 'myself', 'i am', "i'm"],
            'age': ['years old', 'age', 'year old'],
            'school/class': ['school', 'class', 'grade', 'studying'],
            'family': ['family', 'mother', 'father', 'parents', 'siblings', 'brother', 'sister'],
            'hobbies/interest': ['hobby', 'hobbies', 'like', 'enjoy', 'love', 'interest', 'play', 'playing']
        }
        
        for category, keywords in must_have.items():
            if any(kw in text_lower for kw in keywords):
                score += 4
                found_keywords.append(category)
            else:
                missing_keywords.append(category)
        
        # Good to Have keywords (2 points each, max 10)
        good_to_have = {
            'about_family': ['kind', 'caring', 'loving', 'supportive', 'special thing about'],
            'location': ['from', 'live in', 'based in', 'located'],
            'ambition': ['want to', 'dream', 'goal', 'aspire', 'future', 'become'],
            'unique_fact': ['fun fact', 'interesting', 'unique', 'special'],
            'strengths': ['good at', 'strength', 'achievement', 'award', 'excel']
        }
        
        for category, keywords in good_to_have.items():
            if any(kw in text_lower for kw in keywords):
                score += 2
                found_keywords.append(category)
        
        feedback = f"Found: {', '.join(found_keywords)}"
        if missing_keywords:
            feedback += f" | Missing: {', '.join(missing_keywords)}"
        
        return min(score, 30), feedback, found_keywords
    
    def score_flow(self, text):
        """Score flow/structure (0-5 points)"""
        text_lower = text.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 3:
            return 0, "Too short to evaluate flow"
        
        score = 5  # Start with full score
        feedback = []
        
        # Check if salutation is at the beginning
        first_sentence = sentences[0].lower()
        has_salutation = any(word in first_sentence for word in ['hello', 'hi', 'good morning', 'good afternoon', 'good evening'])
        
        # Check if basic details come early (first 3 sentences)
        early_text = ' '.join(sentences[:3]).lower()
        has_early_basics = any(word in early_text for word in ['name', 'myself', 'age', 'class', 'school'])
        
        # Check for closing
        last_sentence = sentences[-1].lower()
        has_closing = any(word in last_sentence for word in ['thank', 'thanks', 'goodbye'])
        
        if not has_salutation:
            score -= 1
            feedback.append("Missing salutation")
        
        if not has_early_basics:
            score -= 2
            feedback.append("Basic details not introduced early")
        
        if not has_closing:
            score -= 1
            feedback.append("No closing statement")
        
        if score == 5:
            feedback = ["Good flow maintained"]
        
        return max(score, 0), "; ".join(feedback)
    
    # ===== SPEECH RATE SCORING =====
    
    def score_speech_rate(self, text, duration_seconds):
        """Score speech rate (0-10 points)"""
        word_count = self.count_words(text)
        wpm = (word_count / duration_seconds) * 60
        
        if wpm > 161:
            score = 2
            category = "Too Fast"
        elif 141 <= wpm <= 160:
            score = 6
            category = "Fast"
        elif 111 <= wpm <= 140:
            score = 10
            category = "Ideal"
        elif 81 <= wpm <= 110:
            score = 6
            category = "Slow"
        else:  # < 80
            score = 2
            category = "Too Slow"
        
        return score, f"{category} ({wpm:.1f} WPM)"
    
    # ===== LANGUAGE & GRAMMAR SCORING =====
    
    def score_grammar(self, text):
        """Score grammar (0-10 points)"""
        word_count = self.count_words(text)
        
        # If LanguageTool is available, use it
        if self.grammar_tool is not None:
            try:
                matches = self.grammar_tool.check(text)
                errors_per_100_words = (len(matches) / word_count) * 100
                grammar_score_ratio = 1 - min(errors_per_100_words / 10, 1)
                
                if grammar_score_ratio >= 0.9:
                    score = 10
                elif grammar_score_ratio >= 0.7:
                    score = 8
                elif grammar_score_ratio >= 0.5:
                    score = 6
                elif grammar_score_ratio >= 0.3:
                    score = 4
                else:
                    score = 2
                
                return score, f"{len(matches)} errors ({errors_per_100_words:.1f} per 100 words)", grammar_score_ratio
            except Exception as e:
                print(f"LanguageTool error: {e}, falling back to basic checks")
        
        # Basic grammar checking (fallback)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        issues = 0
        
        # Check for basic issues
        for sentence in sentences:
            if sentence and not sentence[0].isupper():
                issues += 1
            # Check for double spaces
            if '  ' in sentence:
                issues += 1
        
        errors_per_100_words = (issues / word_count) * 100
        grammar_score_ratio = max(0, 1 - (errors_per_100_words / 10))
        
        if grammar_score_ratio >= 0.9:
            score = 10
        elif grammar_score_ratio >= 0.7:
            score = 8
        elif grammar_score_ratio >= 0.5:
            score = 6
        elif grammar_score_ratio >= 0.3:
            score = 4
        else:
            score = 2
        
        return score, f"Basic check: {issues} issues found", grammar_score_ratio
    
    def score_vocabulary_richness(self, text):
        """Score vocabulary richness using TTR (0-10 points)"""
        words = text.lower().split()
        unique_words = set(words)
        
        ttr = len(unique_words) / len(words) if words else 0
        
        if ttr >= 0.9:
            score = 10
        elif ttr >= 0.7:
            score = 8
        elif ttr >= 0.5:
            score = 6
        elif ttr >= 0.3:
            score = 4
        else:
            score = 2
        
        return score, f"TTR: {ttr:.2f}", ttr
    
    # ===== CLARITY SCORING =====
    
    def score_filler_words(self, text):
        """Score filler word rate (0-15 points)"""
        text_lower = text.lower()
        total_words = self.count_words(text)
        
        filler_count = 0
        found_fillers = []
        
        for filler in self.filler_words:
            count = text_lower.count(f' {filler} ') + text_lower.count(f' {filler},') + text_lower.count(f' {filler}.')
            if count > 0:
                filler_count += count
                found_fillers.append(f"{filler}({count})")
        
        filler_rate = (filler_count / total_words) * 100 if total_words > 0 else 0
        
        if filler_rate <= 3:
            score = 15
        elif filler_rate <= 6:
            score = 12
        elif filler_rate <= 9:
            score = 9
        elif filler_rate <= 12:
            score = 6
        else:
            score = 3
        
        feedback = f"{filler_count} fillers ({filler_rate:.1f}%)"
        if found_fillers:
            feedback += f": {', '.join(found_fillers[:3])}"  # Show first 3
        
        return score, feedback, filler_rate
    
    # ===== ENGAGEMENT SCORING =====
    
    def score_sentiment(self, text):
        """Score sentiment/positivity using VADER (0-15 points)"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        positive_score = sentiment_scores['pos']  # 0 to 1
        
        if positive_score >= 0.9:
            score = 15
        elif positive_score >= 0.7:
            score = 12
        elif positive_score >= 0.5:
            score = 9
        elif positive_score >= 0.3:
            score = 6
        else:
            score = 3
        
        # Determine sentiment category
        compound = sentiment_scores['compound']
        if compound >= 0.05:
            category = "Positive"
        elif compound <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"
        
        return score, f"{category} (score: {positive_score:.2f})", positive_score
    
    # ===== AI FEEDBACK =====
    
    def get_ai_feedback(self, transcript, overall_score, criteria_details):
        """Use Groq API for overall intelligent feedback"""
        prompt = f"""You are evaluating a student's self-introduction transcript. Provide brief, constructive feedback (3-4 sentences).

Transcript: "{transcript}"

Overall Score: {overall_score}/100

Key areas evaluated:
- Content & Structure: {criteria_details.get('content_structure', 'N/A')}
- Speech Rate: {criteria_details.get('speech_rate', 'N/A')}
- Grammar: {criteria_details.get('grammar', 'N/A')}
- Clarity: {criteria_details.get('clarity', 'N/A')}
- Engagement: {criteria_details.get('engagement', 'N/A')}

Provide encouraging, specific, and actionable feedback."""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=200,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Great effort on your self-introduction! Your score of {overall_score}/100 shows promise. Focus on the areas highlighted in the detailed breakdown to improve further."
    
    # ===== MAIN SCORING FUNCTION =====
    
    def score_transcript(self, transcript, duration_seconds=None):
        """Main scoring function following Nirmaan rubric"""
        
        # Calculate basic metrics
        word_count = self.count_words(transcript)
        sentence_count = self.count_sentences(transcript)
        
        # If duration not provided, estimate (average 150 WPM)
        if duration_seconds is None:
            duration_seconds = (word_count / 150) * 60
        
        criteria_results = []
        
        # 1. CONTENT & STRUCTURE (40 points + 10 semantic bonus = 50 total)
        sal_score, sal_feedback = self.score_salutation(transcript)
        kw_score, kw_feedback, kw_found = self.score_keyword_presence(transcript)
        flow_score, flow_feedback = self.score_flow(transcript)
        sem_score, sem_feedback, avg_sim, max_sim = self.score_semantic_similarity(transcript)
        
        content_structure_score = sal_score + kw_score + flow_score + sem_score
        
        criteria_results.append({
            'criterion': 'Content & Structure',
            'total_score': int(content_structure_score),
            'max_score': 50,
            'weight': 40,
            'subcriteria': [
                {'name': 'Salutation', 'score': int(sal_score), 'max': 5, 'feedback': sal_feedback},
                {'name': 'Keyword Presence', 'score': int(kw_score), 'max': 30, 'feedback': kw_feedback},
                {'name': 'Flow', 'score': int(flow_score), 'max': 5, 'feedback': flow_feedback},
                {'name': 'Semantic Similarity (NLP)', 'score': int(sem_score), 'max': 10, 'feedback': sem_feedback}
            ]
        })
        
        # 2. SPEECH RATE (10 points)
        sr_score, sr_feedback = self.score_speech_rate(transcript, duration_seconds)
        
        criteria_results.append({
            'criterion': 'Speech Rate',
            'total_score': int(sr_score),
            'max_score': 10,
            'weight': 10,
            'feedback': sr_feedback
        })
        
        # 3. LANGUAGE & GRAMMAR (20 points)
        gram_score, gram_feedback, gram_ratio = self.score_grammar(transcript)
        vocab_score, vocab_feedback, ttr = self.score_vocabulary_richness(transcript)
        
        language_grammar_score = gram_score + vocab_score
        
        criteria_results.append({
            'criterion': 'Language & Grammar',
            'total_score': int(language_grammar_score),
            'max_score': 20,
            'weight': 20,
            'subcriteria': [
                {'name': 'Grammar', 'score': int(gram_score), 'max': 10, 'feedback': gram_feedback},
                {'name': 'Vocabulary Richness', 'score': int(vocab_score), 'max': 10, 'feedback': vocab_feedback}
            ]
        })
        
        # 4. CLARITY (15 points)
        filler_score, filler_feedback, filler_rate = self.score_filler_words(transcript)
        
        criteria_results.append({
            'criterion': 'Clarity',
            'total_score': int(filler_score),
            'max_score': 15,
            'weight': 15,
            'feedback': filler_feedback
        })
        
        # 5. ENGAGEMENT (15 points)
        sent_score, sent_feedback, pos_score = self.score_sentiment(transcript)
        
        criteria_results.append({
            'criterion': 'Engagement',
            'total_score': int(sent_score),
            'max_score': 15,
            'weight': 15,
            'feedback': sent_feedback
        })
        
        # Calculate overall score (out of 110 now, will normalize to 100)
        total_score = (content_structure_score + sr_score + language_grammar_score + 
                      filler_score + sent_score)
        
        # Normalize to 100 (since max is now 110)
        max_possible = 110
        normalized_score = (total_score / max_possible) * 100
        
        # Get AI feedback
        criteria_summary = {
            'content_structure': f"{content_structure_score}/40",
            'speech_rate': sr_feedback,
            'grammar': gram_feedback,
            'clarity': filler_feedback,
            'engagement': sent_feedback
        }
        
        ai_feedback = self.get_ai_feedback(transcript, normalized_score, criteria_summary)
        
        return {
            'overall_score': float(round(normalized_score, 2)),
            'max_score': 100,
            'words': int(word_count),
            'sentences': int(sentence_count),
            'duration_seconds': float(duration_seconds),
            'criteria_scores': criteria_results,
            'ai_feedback': ai_feedback,
            'semantic_analysis': {
                'avg_similarity': float(round(avg_sim, 3)),
                'max_similarity': float(round(max_sim, 3))
            }
        }