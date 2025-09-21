from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import yt_dlp
import whisper
from transformers import pipeline
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import csv
import json
from yt_dlp.utils import DownloadError
import logging
import tempfile
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Change this in production

# Global variables to store models (load once)
whisper_model = None
summarizer = None

def load_models():
    """Load models once at startup"""
    global whisper_model, summarizer
    try:
        if whisper_model is None:
            logger.info("Loading Whisper model...")
            whisper_model = whisper.load_model("base")  # Use base instead of small for better performance
            logger.info("Whisper model loaded successfully")
        
        # Skip loading summarizer at startup to avoid dependency issues
        # Will load on demand instead
        
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        raise

def get_summarizer():
    """Get summarizer model, loading it if necessary"""
    global summarizer
    if summarizer is None:
        try:
            logger.info("Loading summarization model...")
            from transformers import pipeline
            summarizer = pipeline("summarization", 
                                model="facebook/bart-large-cnn",
                                device=-1)  # Force CPU usage
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load summarization model: {str(e)}")
            return None
    return summarizer

# Initialize Whisper model at startup (lighter weight)
try:
    load_models()
    logger.info("Whisper model initialized successfully at startup")
except Exception as e:
    logger.error(f"Failed to initialize Whisper model at startup: {str(e)}")
    # Don't exit here, let the app start and show error when needed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        
        # Ensure models are loaded
        try:
            load_models()
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            flash(f"AI models failed to load: {str(e)}. Please restart the application.", 'error')
            return render_template('index.html')
        
        # Create a temporary directory for this session
        temp_dir = tempfile.mkdtemp()
        
        try:
            logger.info(f"Processing URL: {url}")
            
            # Load models if not already loaded
            load_models()
            
            # Download audio with better error handling
            audio_file = os.path.join(temp_dir, 'audio')
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': audio_file,
                'quiet': False,  # Enable output for debugging
                'no_warnings': False,
                'extractaudio': True,
                'audioformat': 'wav',  # Convert to WAV for better Whisper compatibility
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'geo_bypass': True,
                'duration_limit': 1800,  # Limit to 30 minutes
                'retries': 3,
                'fragment_retries': 3,
                'http_chunk_size': 10485760,  # 10MB chunks
            }
            
            logger.info("Starting video download...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(url, download=True)
                except Exception as e:
                    logger.error(f"Download failed: {str(e)}")
                    raise DownloadError(f"Failed to download video: {str(e)}")
            
            # Find the downloaded audio file
            audio_files = [f for f in os.listdir(temp_dir) if f.startswith('audio')]
            if not audio_files:
                logger.error("No audio file found after download")
                raise Exception("No audio file found after download")
            
            actual_audio_file = os.path.join(temp_dir, audio_files[0])
            file_size = os.path.getsize(actual_audio_file)
            logger.info(f"Downloaded audio file: {actual_audio_file}, size: {file_size} bytes")
            
            if file_size == 0:
                raise Exception("Audio file is empty")
            
            # Extract metadata
            metadata = {
                'title': info.get('title', 'N/A'),
                'uploader': info.get('uploader', 'N/A'),
                'upload_date': info.get('upload_date', 'N/A'),
                'duration': info.get('duration', 'N/A'),
                'view_count': info.get('view_count', 'N/A'),
                'description': info.get('description', 'N/A')[:500] + '...' if info.get('description', '') else 'N/A'
            }
            
            # Transcribe with timestamps
            logger.info("Starting transcription...")
            try:
                result = whisper_model.transcribe(
                    actual_audio_file,
                    task='transcribe',
                    language='en',  # Specify language for better performance
                    word_timestamps=True,
                    verbose=True
                )
                transcript = result['text'].strip()
                segments = result.get('segments', [])
                
                if not transcript:
                    raise Exception("Transcription resulted in empty text")
                
                logger.info(f"Transcription completed: {len(transcript)} characters, {len(segments)} segments")
                
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                raise Exception(f"Transcription failed: {str(e)}")
            
            # Generate summary
            logger.info("Starting summarization...")
            try:
                # Split long transcripts into chunks
                max_chunk_length = 1024
                chunks = [transcript[i:i+max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]
                summaries = []
                
                for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
                    if len(chunk.strip()) > 50:  # Only process meaningful chunks
                        try:
                            summary_result = summarizer(
                                chunk,
                                max_length=150,
                                min_length=30,
                                do_sample=False,
                                truncation=True
                            )
                            summaries.append(summary_result[0]['summary_text'])
                            logger.info(f"Processed chunk {i+1}/{len(chunks[:3])}")
                        except Exception as chunk_error:
                            logger.warning(f"Failed to summarize chunk {i+1}: {str(chunk_error)}")
                            continue
                
                if summaries:
                    summary = " ".join(summaries)
                else:
                    # Fallback summary
                    summary = transcript[:300] + "..." if len(transcript) > 300 else transcript
                
                logger.info(f"Summarization completed: {len(summary)} characters")
                
            except Exception as e:
                logger.warning(f"Summarization failed: {str(e)}")
                summary = transcript[:300] + "..." if len(transcript) > 300 else transcript
            
            # Generate highlights (improved logic)
            logger.info("Generating highlights...")
            highlights = []
            if segments:
                # Sort by segment length and take top 5
                long_segments = sorted(segments, key=lambda x: len(x['text']), reverse=True)[:5]
                highlights = [{
                    'text': seg['text'].strip(),
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'clip_id': f"highlight_{i+1}"
                } for i, seg in enumerate(long_segments) if len(seg['text'].strip()) > 20]
            
            # Generate quizzes (enhanced with 20 comprehensive questions)
            logger.info("Generating enhanced quizzes (20 questions)...")
            quizzes = []
            try:
                # Extract comprehensive data for better quiz generation
                sentences = [s.strip() for s in transcript.split('.') if len(s.strip()) > 15][:30]
                
                # Enhanced keywords extraction
                words = transcript.lower().split()
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'also', 'just', 'so', 'very', 'really', 'now', 'get', 'make', 'go', 'come', 'know', 'see', 'want', 'like', 'think', 'say', 'said', 'one', 'two', 'three', 'first', 'last', 'next', 'then', 'when', 'where', 'how', 'what', 'why', 'who', 'which'}
                
                # Extract keywords with frequency
                word_freq = {}
                for word in words:
                    if len(word) > 4 and word not in common_words and word.isalpha():
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top keywords by frequency
                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
                keywords = [word for word, freq in top_keywords]
                
                # Extract numbers and dates mentioned
                import re
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', transcript)
                dates = re.findall(r'\b\d{4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', transcript.lower())
                
                quiz_count = 0
                max_quizzes = 20

                # 1. Multiple choice questions (5 questions)
                for sentence in sentences[:8]:
                    if quiz_count >= 5:
                        break
                    if len(sentence.split()) > 12:
                        words_in_sentence = sentence.split()
                        # Find meaningful words to replace
                        meaningful_words = [w for w in words_in_sentence if len(w) > 4 and w.lower() not in common_words]
                        
                        if meaningful_words:
                            correct_answer = meaningful_words[0]
                            
                            # Generate plausible wrong options
                            wrong_options = []
                            for keyword in keywords[:10]:
                                if keyword != correct_answer.lower() and len(wrong_options) < 3:
                                    wrong_options.append(keyword.capitalize())
                            
                            if len(wrong_options) >= 3:
                                question_text = sentence.replace(correct_answer, "______", 1)
                                options = [correct_answer] + wrong_options[:3]
                                import random
                                random.shuffle(options)
                                
                                quizzes.append({
                                    'question': f"Fill in the blank: {question_text}",
                                    'answer': f"{correct_answer}",
                                    'type': 'multiple-choice',
                                    'options': options
                                })
                                quiz_count += 1

                # 2. True/False questions (6 questions)
                true_false_sources = sentences + summary.split('.')
                for sent in true_false_sources[:12]:
                    if quiz_count >= 11:
                        break
                    if len(sent.strip()) > 25:
                        # True statement
                        quizzes.append({
                            'question': f"True or False: {sent.strip()}",
                            'answer': "True",
                            'type': 'true-false'
                        })
                        quiz_count += 1
                        
                        # False statement (modify the sentence)
                        if quiz_count < 11:
                            words = sent.split()
                            if len(words) > 6:
                                # Create false version by negating or changing key words
                                if 'is' in words:
                                    false_sent = sent.replace(' is ', ' is not ')
                                elif 'are' in words:
                                    false_sent = sent.replace(' are ', ' are not ')
                                elif 'will' in words:
                                    false_sent = sent.replace(' will ', ' will not ')
                                else:
                                    false_sent = sent + " (This is incorrect)"
                                
                                quizzes.append({
                                    'question': f"True or False: {false_sent.strip()}",
                                    'answer': "False",
                                    'type': 'true-false'
                                })
                                quiz_count += 1

                # 3. Short answer questions about key concepts (5 questions)
                for keyword in keywords[:8]:
                    if quiz_count >= 16:
                        break
                    # Find sentences containing this keyword
                    relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]
                    if relevant_sentences:
                        best_sentence = max(relevant_sentences, key=len)  # Get most detailed sentence
                        quizzes.append({
                            'question': f"Explain what was discussed about '{keyword.capitalize()}' in the video.",
                            'answer': best_sentence[:250] + "..." if len(best_sentence) > 250 else best_sentence,
                            'type': 'short-answer'
                        })
                        quiz_count += 1

                # 4. Numerical/factual questions (2 questions)
                if numbers and quiz_count < 18:
                    for num in numbers[:3]:
                        if quiz_count >= 18:
                            break
                        # Find context for this number
                        num_contexts = [s for s in sentences if num in s]
                        if num_contexts:
                            context = num_contexts[0]
                            quizzes.append({
                                'question': f"What significance does the number '{num}' have according to the video?",
                                'answer': context[:200] + "..." if len(context) > 200 else context,
                                'type': 'factual'
                            })
                            quiz_count += 1

                # 5. Comprehension and analysis questions (2 questions)
                if quiz_count < 20:
                    quizzes.append({
                        'question': "What is the main theme or central message of this video?",
                        'answer': summary[:200] + "..." if len(summary) > 200 else summary,
                        'type': 'comprehension'
                    })
                    quiz_count += 1
                
                if quiz_count < 20:
                    quizzes.append({
                        'question': "What are the key takeaways or conclusions from this video?",
                        'answer': f"Based on the content, the main points include: {', '.join(keywords[:5])} and the overall message focuses on {summary.split('.')[0]}",
                        'type': 'analysis'
                    })
                    quiz_count += 1

                # Fill remaining slots with keyword-based questions
                remaining_questions = [
                    "What specific examples or case studies were mentioned?",
                    "What problems or challenges were discussed?", 
                    "What solutions or recommendations were provided?",
                    "What evidence or data was presented to support the main points?",
                    "How does this topic relate to current trends or issues?"
                ]
                
                for i, q in enumerate(remaining_questions):
                    if quiz_count >= 20:
                        break
                    relevant_content = " ".join(sentences[i*2:(i*2)+3]) if len(sentences) > i*2+2 else summary
                    quizzes.append({
                        'question': q,
                        'answer': relevant_content[:250] + "..." if len(relevant_content) > 250 else relevant_content,
                        'type': 'analytical'
                    })
                    quiz_count += 1

                # Ensure we have exactly 20 questions
                while len(quizzes) < 20:
                    quizzes.append({
                        'question': f"Additional Question {len(quizzes) + 1}: What other important points were covered in this video?",
                        'answer': "Please review the transcript and summary for additional details.",
                        'type': 'open-ended'
                    })

                quizzes = quizzes[:20]  # Ensure exactly 20 questions
                logger.info(f"Generated {len(quizzes)} comprehensive quiz questions")

            except Exception as e:
                logger.warning(f"Enhanced quiz generation failed: {str(e)}")
                # Fallback to basic questions
                quizzes = []
                for i in range(20):
                    quizzes.append({
                        'question': f'Question {i+1}: What key point was discussed in this video?',
                        'answer': f'Based on the content: {summary[:100] if summary else "Please refer to the transcript"}',
                        'type': 'general'
                    })
            
            # Generate output files
            output_files = {}
            
            # Text report
            logger.info("Generating text report...")
            try:
                text_report = f"""VIDEO ANALYSIS REPORT
{'='*50}

METADATA:
{json.dumps(metadata, indent=2)}

SUMMARY:
{summary}

FULL TRANSCRIPT:
{transcript}

HIGHLIGHTS:
{'='*20}
"""
                for h in highlights:
                    text_report += f"{h['clip_id']}: {h['text']} (Time: {h['start']:.1f}s - {h['end']:.1f}s)\n\n"
                
                text_report += "\nQUIZ QUESTIONS:\n" + "="*20 + "\n"
                for i, q in enumerate(quizzes, 1):
                    text_report += f"{i}. {q['question']} (Type: {q['type']})\n   Answer: {q['answer']}\n\n"
                
                text_output = "report.txt"
                with open(text_output, 'w', encoding='utf-8') as f:
                    f.write(text_report)
                output_files['text'] = text_output
                logger.info("Text report generated successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate text report: {str(e)}")
                output_files['text'] = None
            
            # PDF report
            logger.info("Generating PDF report...")
            try:
                pdf_output = "video_analysis.pdf"
                doc = SimpleDocTemplate(pdf_output, pagesize=letter)
                styles = getSampleStyleSheet()
                
                # Create custom styles
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=16,
                    spaceAfter=30,
                    alignment=1  # Center alignment
                )
                
                heading_style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading2'],
                    fontSize=12,
                    spaceAfter=15,
                    textColor=colors.blue
                )
                
                normal_style = styles['Normal']
                normal_style.fontSize = 10
                
                # Build the PDF content
                content = []
                
                # Title
                content.append(Paragraph("Video Analysis Report", title_style))
                content.append(Spacer(1, 12))
                
                # Metadata section
                content.append(Paragraph("Video Metadata:", heading_style))
                metadata_data = [
                    ["Title", str(metadata.get('title', 'N/A'))],
                    ["Channel", str(metadata.get('uploader', 'N/A'))],
                    ["Duration", f"{metadata.get('duration', 'N/A')} seconds"],
                    ["Views", str(metadata.get('view_count', 'N/A'))],
                    ["Upload Date", str(metadata.get('upload_date', 'N/A'))],
                    ["Description", str(metadata.get('description', 'N/A'))[:200] + "..."]
                ]
                
                metadata_table = Table(metadata_data, colWidths=[100, 300])
                metadata_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                content.append(metadata_table)
                content.append(Spacer(1, 20))
                
                # Summary section
                content.append(Paragraph("Summary:", heading_style))
                content.append(Paragraph(summary, normal_style))
                content.append(Spacer(1, 20))
                
                # Highlights section
                if highlights:
                    content.append(Paragraph("Key Highlights:", heading_style))
                    for h in highlights[:5]:  # Limit to first 5
                        highlight_text = f"{h['clip_id']}: {h['text']} (Time: {h['start']:.1f}s - {h['end']:.1f}s)"
                        content.append(Paragraph(highlight_text, normal_style))
                        content.append(Spacer(1, 10))
                    content.append(Spacer(1, 10))
                
                # Quiz section
                if quizzes:
                    content.append(Paragraph("Generated Quiz Questions:", heading_style))
                    for i, q in enumerate(quizzes, 1):
                        quiz_text = f"{i}. {q['question']} (Type: {q['type']})<br/>Answer: {q['answer']}"
                        content.append(Paragraph(quiz_text, normal_style))
                        content.append(Spacer(1, 10))
                
                # Build the PDF
                doc.build(content)
                output_files['pdf'] = pdf_output
                logger.info("PDF report generated successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate PDF: {str(e)}")
                output_files['pdf'] = None
            
            # Generate professional text report PDF (includes full transcript)
            logger.info("Generating professional text report PDF...")
            try:
                text_pdf_output = "complete_report.pdf"
                doc = SimpleDocTemplate(text_pdf_output, pagesize=letter,
                                      leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
                styles = getSampleStyleSheet()

                # Create professional styles with enhanced formatting
                title_style = ParagraphStyle(
                    'ReportTitle',
                    parent=styles['Heading1'],
                    fontSize=20,
                    spaceAfter=25,
                    alignment=1,  # Center
                    textColor=colors.darkblue,
                    fontName='Helvetica-Bold',
                    borderColor=colors.darkblue,
                    borderWidth=2,
                    borderPadding=10
                )

                subtitle_style = ParagraphStyle(
                    'ReportSubtitle',
                    parent=styles['Normal'],
                    fontSize=12,
                    spaceAfter=30,
                    alignment=1,
                    textColor=colors.darkgrey,
                    fontName='Helvetica-Oblique'
                )

                section_style = ParagraphStyle(
                    'SectionHeader',
                    parent=styles['Heading2'],
                    fontSize=16,
                    spaceAfter=18,
                    textColor=colors.white,
                    fontName='Helvetica-Bold',
                    backgroundColor=colors.darkblue,
                    borderColor=colors.darkblue,
                    borderWidth=1,
                    borderPadding=8,
                    leftIndent=0
                )

                subsection_style = ParagraphStyle(
                    'SubSectionHeader',
                    parent=styles['Heading3'],
                    fontSize=14,
                    spaceAfter=12,
                    textColor=colors.darkgreen,
                    fontName='Helvetica-Bold'
                )

                normal_style = ParagraphStyle(
                    'NormalText',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=10,
                    alignment=0,  # Left align
                    leading=14
                )

                transcript_style = ParagraphStyle(
                    'TranscriptText',
                    parent=styles['Normal'],
                    fontSize=10,
                    spaceAfter=8,
                    leftIndent=15,
                    alignment=0,
                    leading=12,
                    textColor=colors.darkslategray
                )

                highlight_style = ParagraphStyle(
                    'HighlightText',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=8,
                    leftIndent=10,
                    backgroundColor=colors.lightyellow,
                    borderColor=colors.orange,
                    borderWidth=1,
                    borderPadding=5
                )

                quiz_style = ParagraphStyle(
                    'QuizText',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=6,
                    leftIndent=20,
                    textColor=colors.darkblue
                )

                answer_style = ParagraphStyle(
                    'AnswerText',
                    parent=styles['Normal'],
                    fontSize=10,
                    spaceAfter=12,
                    leftIndent=40,
                    textColor=colors.darkgreen,
                    fontName='Helvetica-Bold'
                )

                # Build professional PDF content
                content = []

                # Title page with enhanced design
                content.append(Paragraph("🎥 Video Analysis Report", title_style))
                content.append(Spacer(1, 15))
                content.append(Paragraph("Complete Professional Analysis", subtitle_style))

                # Generation info with better formatting
                import datetime
                gen_time = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
                content.append(Paragraph(f"Report Generated: {gen_time}", ParagraphStyle(
                    'GenInfo',
                    parent=styles['Normal'],
                    fontSize=10,
                    alignment=1,
                    textColor=colors.grey,
                    spaceAfter=40
                )))

                # Table of Contents
                content.append(Paragraph("Table of Contents", section_style))
                toc_data = [
                    ["1. Video Information", "Page 2"],
                    ["2. Executive Summary", "Page 3"],
                    ["3. Key Highlights", "Page 4"],
                    ["4. Assessment Questions", "Page 5"],
                    ["5. Complete Transcript", "Page 6"]
                ]

                toc_table = Table(toc_data, colWidths=[350, 100])
                toc_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                content.append(toc_table)
                content.append(Spacer(1, 30))

                # Page break before main content
                content.append(Spacer(1, 50))

                # 1. Video Information section
                content.append(Paragraph("1. Video Information", section_style))
                content.append(Spacer(1, 15))

                # Enhanced metadata table
                metadata_data = [
                    ["📹 Title", str(metadata.get('title', 'N/A'))],
                    ["👤 Channel/Uploader", str(metadata.get('uploader', 'N/A'))],
                    ["⏱️ Duration", f"{metadata.get('duration', 'N/A')} seconds"],
                    ["👁️ View Count", f"{metadata.get('view_count', 'N/A'):,}" if metadata.get('view_count') else 'N/A'],
                    ["📅 Upload Date", str(metadata.get('upload_date', 'N/A'))],
                    ["📝 Description", str(metadata.get('description', 'N/A'))[:300] + "..." if len(str(metadata.get('description', ''))) > 300 else str(metadata.get('description', 'N/A'))]
                ]

                metadata_table = Table(metadata_data, colWidths=[140, 330])
                metadata_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
                ]))
                content.append(metadata_table)
                content.append(Spacer(1, 25))

                # 2. Executive Summary section
                content.append(Paragraph("2. Executive Summary", section_style))
                content.append(Spacer(1, 15))

                # Split summary into paragraphs for better formatting
                summary_paragraphs = summary.split('\n')
                for para in summary_paragraphs:
                    if para.strip():
                        content.append(Paragraph(para.strip(), normal_style))
                        content.append(Spacer(1, 6))
                content.append(Spacer(1, 20))

                # 3. Key Highlights section
                if highlights:
                    content.append(Paragraph("3. Key Highlights", section_style))
                    content.append(Spacer(1, 15))

                    for i, h in enumerate(highlights, 1):
                        # Highlight header
                        content.append(Paragraph(f"Highlight {i}", subsection_style))

                        # Highlight text in styled box
                        highlight_text = h['text'].strip()
                        content.append(Paragraph(highlight_text, highlight_style))

                        # Timestamp info
                        timestamp_text = f"⏰ Timestamp: {h['start']:.1f}s - {h['end']:.1f}s"
                        content.append(Paragraph(timestamp_text, ParagraphStyle(
                            'Timestamp',
                            parent=styles['Normal'],
                            fontSize=9,
                            textColor=colors.grey,
                            leftIndent=10,
                            spaceAfter=15
                        )))
                        content.append(Spacer(1, 8))

                    content.append(Spacer(1, 20))

                # 4. Assessment Questions section
                if quizzes:
                    content.append(Paragraph("4. Assessment Questions", section_style))
                    content.append(Spacer(1, 15))

                    for i, q in enumerate(quizzes, 1):
                        # Question number and type
                        question_header = f"Question {i} - {q['type'].replace('-', ' ').title()}"
                        content.append(Paragraph(question_header, subsection_style))

                        # Question text
                        content.append(Paragraph(q['question'], quiz_style))
                        content.append(Spacer(1, 5))

                        # Answer
                        content.append(Paragraph(f"Answer: {q['answer']}", answer_style))
                        content.append(Spacer(1, 15))

                    content.append(Spacer(1, 20))

                # 5. Complete Transcript section
                content.append(Paragraph("5. Complete Transcript", section_style))
                content.append(Spacer(1, 15))

                # Introductory note
                content.append(Paragraph("The following is the complete transcribed text from the video:", ParagraphStyle(
                    'TranscriptIntro',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.grey,
                    spaceAfter=15,
                    fontName='Helvetica-Oblique'
                )))

                # Split transcript into manageable paragraphs
                transcript_paragraphs = []
                current_para = ""
                sentences = transcript.split('.')

                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        if len(current_para + sentence) < 800:  # Keep paragraphs reasonable length
                            current_para += sentence + ". "
                        else:
                            if current_para.strip():
                                transcript_paragraphs.append(current_para.strip())
                            current_para = sentence + ". "

                if current_para.strip():
                    transcript_paragraphs.append(current_para.strip())

                for para in transcript_paragraphs:
                    content.append(Paragraph(para, transcript_style))
                    content.append(Spacer(1, 6))

                # Footer with page numbers and generation info
                content.append(Spacer(1, 40))
                content.append(Paragraph("--- End of Report ---", ParagraphStyle(
                    'ReportFooter',
                    parent=styles['Normal'],
                    fontSize=9,
                    alignment=1,
                    textColor=colors.grey,
                    spaceAfter=10
                )))

                content.append(Paragraph(f"Generated by Video Summarizer AI on {gen_time}", ParagraphStyle(
                    'FooterDetails',
                    parent=styles['Normal'],
                    fontSize=8,
                    alignment=1,
                    textColor=colors.darkgrey
                )))

                # Build the professional PDF
                doc.build(content)
                output_files['text_pdf'] = text_pdf_output
                logger.info("Professional text report PDF generated successfully")

            except Exception as e:
                logger.error(f"Failed to generate professional text PDF: {str(e)}")
                output_files['text_pdf'] = None
            
            # Generate DETAILED REPORT with enhanced data extraction and highlighted transcript
            logger.info("Generating detailed comprehensive report...")
            try:
                # Enhanced data extraction
                logger.info("Performing comprehensive data extraction...")
                
                # Extract more comprehensive data
                detailed_data = {}
                
                # 1. Enhanced keyword extraction with frequency and context
                import re
                word_freq = {}
                for word in transcript.lower().split():
                    clean_word = re.sub(r'[^a-zA-Z]', '', word)
                    if len(clean_word) > 4:
                        word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
                
                top_keywords_detailed = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
                detailed_data['keywords'] = [{'word': word, 'frequency': freq} for word, freq in top_keywords_detailed]
                
                # 2. Extract entities (names, places, organizations)
                entity_patterns = {
                    'names': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                    'organizations': r'\b(?:Company|Corporation|Inc|LLC|Ltd|University|Institute|Foundation|Association)\b',
                    'locations': r'\b(?:New York|California|London|Paris|Tokyo|USA|UK|Canada|Australia)\b',
                    'dates': r'\b(?:\d{1,2}\/\d{1,2}\/\d{4}|\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                    'numbers': r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',
                    'percentages': r'\b\d+(?:\.\d+)?%\b',
                    'currencies': r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'
                }
                
                detailed_data['entities'] = {}
                for entity_type, pattern in entity_patterns.items():
                    matches = re.findall(pattern, transcript, re.IGNORECASE)
                    detailed_data['entities'][entity_type] = list(set(matches))[:10]  # Top 10 unique matches
                
                # 3. Sentiment analysis (simple approach)
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'perfect', 'best', 'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'success', 'effective', 'beneficial', 'valuable', 'important', 'useful', 'helpful']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'problem', 'issue', 'difficult', 'challenge', 'failure', 'error', 'mistake', 'wrong', 'disappointing', 'concerning']
                
                transcript_lower = transcript.lower()
                positive_count = sum(transcript_lower.count(word) for word in positive_words)
                negative_count = sum(transcript_lower.count(word) for word in negative_words)
                
                if positive_count + negative_count > 0:
                    sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                    if sentiment_score > 0.3:
                        sentiment = "Positive"
                    elif sentiment_score < -0.3:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"
                else:
                    sentiment = "Neutral"
                
                detailed_data['sentiment'] = {
                    'overall': sentiment,
                    'positive_indicators': positive_count,
                    'negative_indicators': negative_count,
                    'score': sentiment_score if positive_count + negative_count > 0 else 0
                }
                
                # 4. Topic extraction (simple clustering of keywords)
                topics = []
                tech_keywords = [word for word in detailed_data['keywords'][:20] if any(tech in word['word'].lower() for tech in ['tech', 'data', 'digital', 'computer', 'software', 'system', 'network', 'internet', 'web', 'app', 'platform'])]
                business_keywords = [word for word in detailed_data['keywords'][:20] if any(biz in word['word'].lower() for biz in ['business', 'market', 'company', 'revenue', 'profit', 'customer', 'sales', 'strategy', 'management', 'finance'])]
                education_keywords = [word for word in detailed_data['keywords'][:20] if any(edu in word['word'].lower() for edu in ['learn', 'teach', 'education', 'student', 'school', 'university', 'course', 'study', 'knowledge', 'training'])]
                
                if tech_keywords:
                    topics.append({'category': 'Technology', 'keywords': [kw['word'] for kw in tech_keywords]})
                if business_keywords:
                    topics.append({'category': 'Business', 'keywords': [kw['word'] for kw in business_keywords]})
                if education_keywords:
                    topics.append({'category': 'Education', 'keywords': [kw['word'] for kw in education_keywords]})
                
                detailed_data['topics'] = topics
                
                # 5. Text statistics
                detailed_data['statistics'] = {
                    'total_words': len(transcript.split()),
                    'total_sentences': len([s for s in transcript.split('.') if s.strip()]),
                    'average_sentence_length': len(transcript.split()) / max(len([s for s in transcript.split('.') if s.strip()]), 1),
                    'unique_words': len(set(transcript.lower().split())),
                    'readability_score': len(transcript) / max(len(transcript.split()), 1)  # Simple readability measure
                }
                
                # Generate detailed PDF report
                detailed_pdf_output = "detailed_analysis_report.pdf"
                doc = SimpleDocTemplate(detailed_pdf_output, pagesize=letter,
                                      leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
                styles = getSampleStyleSheet()
                
                # Enhanced styles for detailed report
                title_style = ParagraphStyle(
                    'DetailedTitle',
                    parent=styles['Heading1'],
                    fontSize=22,
                    spaceAfter=30,
                    alignment=1,
                    textColor=colors.darkred,
                    fontName='Helvetica-Bold',
                    borderColor=colors.darkred,
                    borderWidth=3,
                    borderPadding=15
                )
                
                section_style = ParagraphStyle(
                    'DetailedSection',
                    parent=styles['Heading2'],
                    fontSize=16,
                    spaceAfter=15,
                    textColor=colors.white,
                    fontName='Helvetica-Bold',
                    backgroundColor=colors.darkred,
                    borderColor=colors.darkred,
                    borderWidth=1,
                    borderPadding=10
                )
                
                subsection_style = ParagraphStyle(
                    'DetailedSubsection',
                    parent=styles['Heading3'],
                    fontSize=14,
                    spaceAfter=10,
                    textColor=colors.darkblue,
                    fontName='Helvetica-Bold',
                    leftIndent=10
                )
                
                highlight_style = ParagraphStyle(
                    'HighlightedText',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=8,
                    backgroundColor=colors.yellow,
                    borderColor=colors.orange,
                    borderWidth=1,
                    borderPadding=8,
                    leftIndent=10,
                    rightIndent=10
                )
                
                keyword_style = ParagraphStyle(
                    'KeywordHighlight',
                    parent=styles['Normal'],
                    fontSize=10,
                    backgroundColor=colors.lightblue,
                    textColor=colors.darkblue,
                    fontName='Helvetica-Bold'
                )
                
                # Build detailed report content
                content = []
                
                # Title page
                content.append(Paragraph("📊 DETAILED VIDEO ANALYSIS REPORT", title_style))
                content.append(Spacer(1, 20))
                content.append(Paragraph("Comprehensive Data-Driven Analysis", ParagraphStyle(
                    'DetailedSubtitle',
                    parent=styles['Normal'],
                    fontSize=14,
                    alignment=1,
                    textColor=colors.darkgrey,
                    fontName='Helvetica-Oblique',
                    spaceAfter=40
                )))
                
                # Generation timestamp
                import datetime
                gen_time = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
                content.append(Paragraph(f"Generated: {gen_time}", ParagraphStyle(
                    'GenTime',
                    parent=styles['Normal'],
                    fontSize=10,
                    alignment=1,
                    textColor=colors.grey,
                    spaceAfter=50
                )))
                
                # Executive Dashboard
                content.append(Paragraph("📈 EXECUTIVE DASHBOARD", section_style))
                content.append(Spacer(1, 15))
                
                dashboard_data = [
                    ["📹 Title", str(metadata.get('title', 'N/A'))],
                    ["👤 Channel", str(metadata.get('uploader', 'N/A'))],
                    ["⏱️ Duration", f"{metadata.get('duration', 'N/A')} seconds"],
                    ["👁️ Views", f"{metadata.get('view_count', 'N/A'):,}" if metadata.get('view_count') else 'N/A'],
                    ["📅 Upload Date", str(metadata.get('upload_date', 'N/A'))],
                    ["📝 Total Words", str(detailed_data['statistics']['total_words'])],
                    ["📊 Unique Words", str(detailed_data['statistics']['unique_words'])],
                    ["🎯 Sentiment", detailed_data['sentiment']['overall']],
                    ["🧠 Quiz Questions", str(len(quizzes))],
                    ["⭐ Key Highlights", str(len(highlights))]
                ]
                
                dashboard_table = Table(dashboard_data, colWidths=[150, 320])
                dashboard_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 2, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
                ]))
                content.append(dashboard_table)
                content.append(Spacer(1, 25))
                
                # Comprehensive Data Analysis
                content.append(Paragraph("🔍 COMPREHENSIVE DATA ANALYSIS", section_style))
                content.append(Spacer(1, 15))
                
                # Keywords Analysis
                content.append(Paragraph("Top Keywords by Frequency", subsection_style))
                top_10_keywords = detailed_data['keywords'][:10]
                keyword_text = " | ".join([f"<b>{kw['word']}</b> ({kw['frequency']})" for kw in top_10_keywords])
                content.append(Paragraph(keyword_text, keyword_style))
                content.append(Spacer(1, 15))
                
                # Entities Analysis
                if detailed_data['entities']:
                    content.append(Paragraph("Extracted Entities", subsection_style))
                    for entity_type, entities in detailed_data['entities'].items():
                        if entities:
                            content.append(Paragraph(f"<b>{entity_type.capitalize()}:</b> {', '.join(entities[:5])}", styles['Normal']))
                    content.append(Spacer(1, 15))
                
                # Topics Analysis
                if detailed_data['topics']:
                    content.append(Paragraph("Identified Topics", subsection_style))
                    for topic in detailed_data['topics']:
                        content.append(Paragraph(f"<b>{topic['category']}:</b> {', '.join(topic['keywords'][:5])}", styles['Normal']))
                    content.append(Spacer(1, 15))
                
                # Sentiment Analysis
                content.append(Paragraph("Sentiment Analysis", subsection_style))
                sentiment_text = f"Overall Sentiment: <b>{detailed_data['sentiment']['overall']}</b><br/>"
                sentiment_text += f"Positive Indicators: {detailed_data['sentiment']['positive_indicators']}<br/>"
                sentiment_text += f"Negative Indicators: {detailed_data['sentiment']['negative_indicators']}<br/>"
                sentiment_text += f"Sentiment Score: {detailed_data['sentiment']['score']:.2f}"
                content.append(Paragraph(sentiment_text, styles['Normal']))
                content.append(Spacer(1, 20))
                
                # Enhanced Quiz Section (20 Questions)
                content.append(Paragraph("🧠 COMPREHENSIVE ASSESSMENT (20 Questions)", section_style))
                content.append(Spacer(1, 15))
                
                for i, q in enumerate(quizzes, 1):
                    question_header = f"Q{i}: {q['type'].replace('-', ' ').title()}"
                    content.append(Paragraph(question_header, subsection_style))
                    content.append(Paragraph(q['question'], styles['Normal']))
                    content.append(Paragraph(f"<b>Answer:</b> {q['answer']}", ParagraphStyle(
                        'AnswerText',
                        parent=styles['Normal'],
                        fontSize=10,
                        spaceAfter=12,
                        leftIndent=20,
                        textColor=colors.darkgreen,
                        fontName='Helvetica-Bold'
                    )))
                    content.append(Spacer(1, 10))
                
                content.append(Spacer(1, 20))
                
                # HIGHLIGHTED FULL TRANSCRIPT
                content.append(Paragraph("📝 FULL TRANSCRIPT WITH HIGHLIGHTS", section_style))
                content.append(Spacer(1, 15))
                
                # Highlight important keywords in the transcript
                highlighted_transcript = transcript
                important_keywords = [kw['word'] for kw in detailed_data['keywords'][:15]]
                
                # Create highlighted version of transcript
                highlighted_paragraphs = []
                sentences = transcript.split('.')
                
                for sentence in sentences:
                    if sentence.strip():
                        highlighted_sentence = sentence.strip() + "."
                        
                        # Check if sentence contains important keywords
                        contains_keywords = any(keyword.lower() in sentence.lower() for keyword in important_keywords)
                        
                        if contains_keywords:
                            # Highlight this sentence
                            highlighted_paragraphs.append({
                                'text': highlighted_sentence,
                                'highlight': True
                            })
                        else:
                            highlighted_paragraphs.append({
                                'text': highlighted_sentence,
                                'highlight': False
                            })
                
                # Add highlighted transcript to PDF
                for para in highlighted_paragraphs[:50]:  # Limit to first 50 sentences for PDF size
                    if para['highlight']:
                        content.append(Paragraph(f"🔥 {para['text']}", highlight_style))
                    else:
                        content.append(Paragraph(para['text'], styles['Normal']))
                    content.append(Spacer(1, 6))
                
                # Statistics Summary
                content.append(Spacer(1, 30))
                content.append(Paragraph("📊 STATISTICAL SUMMARY", section_style))
                content.append(Spacer(1, 15))
                
                stats_data = [
                    ["Total Words", str(detailed_data['statistics']['total_words'])],
                    ["Total Sentences", str(detailed_data['statistics']['total_sentences'])],
                    ["Average Sentence Length", f"{detailed_data['statistics']['average_sentence_length']:.1f} words"],
                    ["Unique Words", str(detailed_data['statistics']['unique_words'])],
                    ["Vocabulary Richness", f"{(detailed_data['statistics']['unique_words'] / detailed_data['statistics']['total_words'] * 100):.1f}%"],
                    ["Top Keyword Frequency", str(detailed_data['keywords'][0]['frequency']) if detailed_data['keywords'] else "0"]
                ]
                
                stats_table = Table(stats_data, colWidths=[200, 270])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                content.append(stats_table)
                
                # Footer
                content.append(Spacer(1, 40))
                content.append(Paragraph("--- END OF DETAILED ANALYSIS ---", ParagraphStyle(
                    'DetailedFooter',
                    parent=styles['Normal'],
                    fontSize=12,
                    alignment=1,
                    textColor=colors.darkred,
                    fontName='Helvetica-Bold',
                    spaceAfter=10
                )))
                
                content.append(Paragraph(f"Generated by Advanced Video Analyzer on {gen_time}", ParagraphStyle(
                    'FooterDetails',
                    parent=styles['Normal'],
                    fontSize=8,
                    alignment=1,
                    textColor=colors.darkgrey
                )))
                
                # Build the detailed PDF
                doc.build(content)
                output_files['detailed_pdf'] = detailed_pdf_output
                logger.info("Detailed comprehensive report generated successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate detailed report: {str(e)}")
                output_files['detailed_pdf'] = None
            
            # CSV for quizzes
            logger.info("Generating CSV file...")
            try:
                csv_output = "quiz_questions.csv"
                with open(csv_output, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['question_number', 'question', 'answer', 'type']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for i, quiz in enumerate(quizzes, 1):
                        writer.writerow({
                            'question_number': i,
                            'question': quiz['question'],
                            'answer': quiz['answer'],
                            'type': quiz['type']
                        })
                output_files['csv'] = csv_output
                logger.info("CSV file generated successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate CSV: {str(e)}")
                output_files['csv'] = None
            
            logger.info("Processing completed successfully")
            
            return render_template('index.html', 
                                 success=True,
                                 summary=summary, 
                                 metadata=metadata, 
                                 highlights=highlights, 
                                 quizzes=quizzes,
                                 pdf_file=output_files.get('pdf'),
                                 csv_file=output_files.get('csv'),
                                 text_file=output_files.get('text'),
                                 text_pdf=output_files.get('text_pdf'),
                                 detailed_pdf=output_files.get('detailed_pdf'))
        
        except DownloadError as e:
            logger.error(f"Download error: {str(e)}")
            flash(f"Error downloading video: {str(e)}. Please check the URL and try again.", 'error')
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            flash(f"An error occurred: {str(e)}", 'error')
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                logger.info("Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {str(e)}")
        
        return render_template('index.html')
    
    return render_template('index.html')

@app.route('/download/<file_type>')
def download(file_type):
    """Download generated files"""
    try:
        file_map = {
            'pdf': 'video_analysis.pdf',
            'csv': 'quiz_questions.csv',
            'txt': 'report.txt',
            'text_pdf': 'complete_report.pdf',
            'detailed_pdf': 'detailed_analysis_report.pdf'
        }
        
        filename = file_map.get(file_type)
        if filename and os.path.exists(filename):
            return send_file(filename, as_attachment=True, download_name=filename)
        else:
            flash(f"File not found: {file_type}", 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        flash(f"Error downloading file: {str(e)}", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=False, host='127.0.0.1', port=5000)  # Disable debug mode to prevent reloading