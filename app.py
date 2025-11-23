import streamlit as st
import json
from scorer import CommunicationScorer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Communication Scorer - Nirmaan AI",
    page_icon="🎤",
    layout="wide"
)

# Title
st.title("🎤 AI Communication Skills Scorer")
st.markdown("**Nirmaan AI Intern Case Study** - Analyze and score self-introduction transcripts")

# Sample transcript text
SAMPLE_TRANSCRIPT = """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School.
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself. One thing people don't know about me is that I once stole a toy from one of my cousin.
My favorite subject is science because it is very interesting. Through science I can explore the whole world and make the discoveries and improve the lives of others.
Thank you for listening."""

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Duration input (optional)
    st.subheader("Optional Settings")
    duration_input = st.number_input(
        "Speech Duration (seconds)", 
        min_value=1, 
        max_value=300, 
        value=None,
        help="Leave empty for auto-estimation"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Evaluation Criteria")
    st.markdown("""
    **1. Content & Structure** (40%)
    - Salutation (5pts)
    - Keywords (30pts)
    - Flow (5pts)
    - Semantic Match (10pts bonus)
    
    **2. Speech Rate** (10%)
    - Words per minute
    
    **3. Language & Grammar** (20%)
    - Grammar errors (10pts)
    - Vocabulary richness (10pts)
    
    **4. Clarity** (15%)
    - Filler word rate
    
    **5. Engagement** (15%)
    - Sentiment analysis
    
    ---
    
    **✨ NLP Features:**
    - Semantic similarity analysis
    - Sentence embeddings
    - Context understanding
    """)

# Get API key from environment
groq_api_key = os.getenv('GROQ_API_KEY', '')

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables!")
    st.info("Please add GROQ_API_KEY to your .env file")
    st.stop()

# Initialize scorer (cache it to avoid reloading models)
@st.cache_resource
def load_scorer(api_key):
    return CommunicationScorer(api_key)

try:
    with st.spinner("Loading AI models... (This may take a minute on first run)"):
        scorer = load_scorer(groq_api_key)
    st.success("✅ All systems ready!")
    
    # Initialize session state for transcript
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    
    # Sample text button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📝 Load Sample Text"):
            st.session_state.transcript = SAMPLE_TRANSCRIPT
            st.rerun()  # Force rerun to update the text area
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload Text File"],
        horizontal=True
    )
    
    transcript = ""
    
    if input_method == "Paste Text":
        transcript = st.text_area(
            "Paste your self-introduction transcript here:",
            value=st.session_state.transcript,
            height=250,
            placeholder="Enter the self-introduction transcript...",
            key="transcript_input"
        )
        # Update session state when user types
        st.session_state.transcript = transcript
        
    else:
        uploaded_file = st.file_uploader("Upload transcript file", type=['txt'])
        if uploaded_file:
            transcript = uploaded_file.read().decode('utf-8')
            st.text_area("Transcript content:", value=transcript, height=200, disabled=True)
    
    # Score button
    if st.button("🎯 Score Transcript", type="primary", use_container_width=True):
        if transcript and transcript.strip():
            with st.spinner("Analyzing transcript... Please wait..."):
                # Score the transcript
                results = scorer.score_transcript(
                    transcript, 
                    duration_seconds=duration_input
                )
                
                # Display overall score
                st.markdown("---")
                st.header("📊 Overall Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    score_color = "🟢" if results['overall_score'] >= 70 else "🟡" if results['overall_score'] >= 50 else "🔴"
                    st.metric(
                        label="Overall Score",
                        value=f"{results['overall_score']}/100",
                        delta=f"{score_color}"
                    )
                
                with col2:
                    st.metric(
                        label="Word Count",
                        value=results['words']
                    )
                
                with col3:
                    st.metric(
                        label="Sentences",
                        value=results['sentences']
                    )
                
                with col4:
                    st.metric(
                        label="Duration",
                        value=f"{results['duration_seconds']:.0f}s"
                    )
                
                # Score interpretation
                score = results['overall_score']
                if score >= 80:
                    st.success("🌟 **Excellent!** Outstanding communication skills demonstrated.")
                elif score >= 60:
                    st.info("👍 **Good!** Strong performance with room for minor improvements.")
                elif score >= 40:
                    st.warning("📈 **Fair** - Some good elements, but needs improvement in key areas.")
                else:
                    st.error("🔄 **Needs Work** - Significant improvements needed across multiple areas.")
                
                # AI Feedback
                st.markdown("### 🤖 AI Analysis")
                st.info(results['ai_feedback'])
                
                # Semantic Analysis (if available)
                if 'semantic_analysis' in results:
                    st.markdown("### 🧠 Semantic Similarity Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Average Similarity",
                            value=f"{results['semantic_analysis']['avg_similarity']:.3f}",
                            help="How well the content matches ideal self-introduction patterns"
                        )
                    with col2:
                        st.metric(
                            label="Best Match",
                            value=f"{results['semantic_analysis']['max_similarity']:.3f}",
                            help="Highest similarity with any ideal pattern"
                        )
                    
                    # Visual interpretation
                    avg_sim = results['semantic_analysis']['avg_similarity']
                    if avg_sim >= 0.7:
                        st.success("🎯 Strong semantic alignment with ideal self-introductions!")
                    elif avg_sim >= 0.5:
                        st.info("👍 Good semantic structure detected.")
                    else:
                        st.warning("💡 Consider adding more typical self-introduction elements.")
                
                # Detailed breakdown
                st.markdown("---")
                st.header("📋 Detailed Criterion Breakdown")
                
                for criterion in results['criteria_scores']:
                    # Progress bar for visual representation
                    percentage = (criterion['total_score'] / criterion['max_score']) * 100
                    
                    with st.expander(
                        f"**{criterion['criterion']}** - {criterion['total_score']}/{criterion['max_score']} points ({percentage:.0f}%)",
                        expanded=False
                    ):
                        # Progress bar
                        st.progress(percentage / 100)
                        
                        # Subcriteria if exists
                        if 'subcriteria' in criterion:
                            st.markdown("#### Breakdown:")
                            for sub in criterion['subcriteria']:
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**{sub['name']}:** {sub['feedback']}")
                                with col2:
                                    st.markdown(f"*{sub['score']}/{sub['max']} pts*")
                        else:
                            st.markdown(f"**Feedback:** {criterion.get('feedback', 'N/A')}")
                        
                        # Show percentage contribution to total
                        contribution = (criterion['total_score'] / results['max_score']) * 100
                        st.caption(f"Contributes {contribution:.1f}% to overall score")
                
                # Score summary table
                st.markdown("---")
                st.header("📈 Score Summary")
                
                summary_data = []
                for criterion in results['criteria_scores']:
                    summary_data.append({
                        'Criterion': criterion['criterion'],
                        'Score': f"{criterion['total_score']}/{criterion['max_score']}",
                        'Percentage': f"{(criterion['total_score'] / criterion['max_score']) * 100:.1f}%",
                        'Weight': f"{criterion['weight']}%"
                    })
                
                st.table(summary_data)
                
                # JSON output
                st.markdown("---")
                with st.expander("📄 View JSON Output (for API integration)"):
                    st.json(results)
                
                # Download results
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ Download Results as JSON",
                        data=json.dumps(results, indent=2),
                        file_name="scoring_results.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    # Create a simple text report
                    report = f"""COMMUNICATION SKILLS SCORING REPORT
{'='*50}

Overall Score: {results['overall_score']}/100
Word Count: {results['words']}
Sentences: {results['sentences']}
Duration: {results['duration_seconds']:.0f} seconds

DETAILED SCORES:
{'-'*50}
"""
                    for criterion in results['criteria_scores']:
                        report += f"\n{criterion['criterion']}: {criterion['total_score']}/{criterion['max_score']} ({(criterion['total_score'] / criterion['max_score']) * 100:.1f}%)\n"
                        if 'subcriteria' in criterion:
                            for sub in criterion['subcriteria']:
                                report += f"  - {sub['name']}: {sub['score']}/{sub['max']} - {sub['feedback']}\n"
                        else:
                            report += f"  {criterion.get('feedback', '')}\n"
                    
                    report += f"\n{'='*50}\nAI FEEDBACK:\n{results['ai_feedback']}\n"
                    
                    st.download_button(
                        label="⬇️ Download Text Report",
                        data=report,
                        file_name="scoring_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        else:
            st.warning("⚠️ Please enter a transcript to score!")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    with st.expander("🔍 Show detailed error"):
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built for Nirmaan AI Intern Case Study | "
    "Powered by Groq API + Sentence Transformers</div>",
    unsafe_allow_html=True
)