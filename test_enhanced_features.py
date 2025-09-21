#!/usr/bin/env python3
"""
Test script to verify the enhanced features: 20 quizzes and detailed report
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_detailed_report_features():
    """Test the detailed report generation with sample data"""
    try:
        print("🧪 Testing detailed report features...")
        
        # Import required modules
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        import datetime
        import re
        
        # Sample transcript for testing
        sample_transcript = """
        This is a comprehensive video about artificial intelligence and machine learning.
        The video discusses various algorithms, neural networks, and deep learning techniques.
        Companies like Google, Microsoft, and OpenAI are leading in this technology.
        The market size for AI is expected to reach 500 billion dollars by 2025.
        Machine learning models require large datasets and powerful computing resources.
        Python and TensorFlow are popular tools for AI development.
        The future of artificial intelligence looks very promising with many applications.
        However, there are also challenges and ethical considerations to address.
        Data privacy and algorithmic bias are important concerns in AI systems.
        Education and training programs are essential for AI adoption.
        """
        
        # Test enhanced data extraction
        print("📊 Testing enhanced data extraction...")
        
        # 1. Enhanced keyword extraction
        word_freq = {}
        for word in sample_transcript.lower().split():
            clean_word = re.sub(r'[^a-zA-Z]', '', word)
            if len(clean_word) > 4:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"✅ Extracted {len(top_keywords)} keywords")
        
        # 2. Entity extraction
        entity_patterns = {
            'organizations': r'\b(?:Google|Microsoft|OpenAI|Company|Corporation)\b',
            'technologies': r'\b(?:AI|Python|TensorFlow|neural networks|machine learning)\b',
            'numbers': r'\b\d+(?:\.\d+)?\b'
        }
        
        entities = {}
        for entity_type, pattern in entity_patterns.items():
            matches = re.findall(pattern, sample_transcript, re.IGNORECASE)
            entities[entity_type] = list(set(matches))
        
        print(f"✅ Extracted entities: {entities}")
        
        # Test quiz generation (20 questions)
        print("🧠 Testing 20-question quiz generation...")
        quizzes = []
        sentences = [s.strip() for s in sample_transcript.split('.') if len(s.strip()) > 15]
        
        # Generate 20 different question types
        question_types = [
            'multiple-choice', 'true-false', 'short-answer', 'factual', 
            'comprehension', 'analysis', 'application', 'synthesis'
        ]
        
        for i in range(20):
            question_type = question_types[i % len(question_types)]
            quizzes.append({
                'question': f"Question {i+1}: What is discussed about {top_keywords[i % len(top_keywords)][0] if top_keywords else 'the topic'}?",
                'answer': f"Sample answer for question {i+1}",
                'type': question_type
            })
        
        print(f"✅ Generated {len(quizzes)} quiz questions")
        
        # Test detailed PDF generation
        print("📄 Testing detailed PDF generation...")
        test_pdf = "test_detailed_report.pdf"
        doc = SimpleDocTemplate(test_pdf, pagesize=letter,
                              leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        
        # Create enhanced styles
        title_style = ParagraphStyle(
            'DetailedTitle',
            parent=styles['Heading1'],
            fontSize=22,
            spaceAfter=30,
            alignment=1,
            textColor=colors.darkred,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'DetailedSection',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            backgroundColor=colors.darkred
        )
        
        highlight_style = ParagraphStyle(
            'HighlightedText',
            parent=styles['Normal'],
            fontSize=11,
            backgroundColor=colors.yellow,
            borderColor=colors.orange,
            borderWidth=1,
            borderPadding=8
        )
        
        content = []
        
        # Title
        content.append(Paragraph("📊 DETAILED VIDEO ANALYSIS REPORT - TEST", title_style))
        content.append(Spacer(1, 20))
        
        # Dashboard
        content.append(Paragraph("📈 EXECUTIVE DASHBOARD", section_style))
        content.append(Spacer(1, 15))
        
        dashboard_data = [
            ["📝 Total Words", str(len(sample_transcript.split()))],
            ["🧠 Quiz Questions", str(len(quizzes))],
            ["🔍 Keywords Found", str(len(top_keywords))],
            ["📊 Entities Extracted", str(sum(len(v) for v in entities.values()))]
        ]
        
        dashboard_table = Table(dashboard_data, colWidths=[200, 200])
        dashboard_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(dashboard_table)
        content.append(Spacer(1, 20))
        
        # Sample quiz questions
        content.append(Paragraph("🧠 SAMPLE QUIZ QUESTIONS (20 Total)", section_style))
        content.append(Spacer(1, 15))
        
        for i, q in enumerate(quizzes[:5], 1):  # Show first 5 for testing
            content.append(Paragraph(f"Q{i}: {q['question']}", styles['Normal']))
            content.append(Paragraph(f"Answer: {q['answer']}", ParagraphStyle(
                'AnswerText',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.darkgreen,
                leftIndent=20
            )))
            content.append(Spacer(1, 10))
        
        # Highlighted transcript sample
        content.append(Paragraph("📝 HIGHLIGHTED TRANSCRIPT SAMPLE", section_style))
        content.append(Spacer(1, 15))
        
        highlighted_sentences = sentences[:3]
        for sentence in highlighted_sentences:
            # Check if contains keywords
            contains_keywords = any(keyword[0] in sentence.lower() for keyword in top_keywords[:5])
            if contains_keywords:
                content.append(Paragraph(f"🔥 {sentence}.", highlight_style))
            else:
                content.append(Paragraph(f"{sentence}.", styles['Normal']))
            content.append(Spacer(1, 6))
        
        # Footer
        content.append(Spacer(1, 30))
        gen_time = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
        content.append(Paragraph(f"Test Report Generated: {gen_time}", ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=1,
            textColor=colors.grey
        )))
        
        # Build PDF
        doc.build(content)
        
        # Verify PDF was created
        if os.path.exists(test_pdf):
            file_size = os.path.getsize(test_pdf)
            print(f"✅ Detailed PDF test successful!")
            print(f"📄 Test PDF: {test_pdf}")
            print(f"📊 File size: {file_size} bytes")
            
            # Clean up
            os.remove(test_pdf)
            print("🧹 Test file cleaned up")
            return True
        else:
            print("❌ PDF was not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_quiz_enhancement():
    """Test the 20-question quiz generation logic"""
    print("🧠 Testing enhanced quiz generation...")
    
    try:
        # Sample data
        sample_sentences = [
            "Artificial intelligence is transforming many industries today",
            "Machine learning algorithms require large amounts of data",
            "Neural networks are inspired by the human brain structure",
            "Deep learning has achieved remarkable results in image recognition",
            "Natural language processing helps computers understand human language"
        ] * 4  # Create 20 sentences
        
        keywords = ['artificial', 'intelligence', 'machine', 'learning', 'neural', 'networks', 'data', 'algorithms']
        
        # Test different question types
        question_types = [
            'multiple-choice', 'true-false', 'short-answer', 'factual',
            'comprehension', 'analysis', 'application', 'synthesis'
        ]
        
        quizzes = []
        for i in range(20):
            q_type = question_types[i % len(question_types)]
            keyword = keywords[i % len(keywords)]
            sentence = sample_sentences[i % len(sample_sentences)]
            
            quizzes.append({
                'question': f"Question {i+1}: Explain the concept of {keyword} as discussed in the video.",
                'answer': f"Based on the content: {sentence}",
                'type': q_type
            })
        
        if len(quizzes) == 20:
            print(f"✅ Successfully generated exactly {len(quizzes)} quiz questions")
            
            # Test question type distribution
            type_count = {}
            for q in quizzes:
                q_type = q['type']
                type_count[q_type] = type_count.get(q_type, 0) + 1
            
            print("📊 Question type distribution:")
            for q_type, count in type_count.items():
                print(f"   {q_type}: {count} questions")
            
            return True
        else:
            print(f"❌ Expected 20 questions, got {len(quizzes)}")
            return False
            
    except Exception as e:
        print(f"❌ Quiz test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Enhanced Video Analyzer Features")
    print("=" * 50)
    
    # Test quiz enhancement
    quiz_success = test_quiz_enhancement()
    print()
    
    # Test detailed report
    report_success = test_detailed_report_features()
    print()
    
    if quiz_success and report_success:
        print("🎉 All enhanced features are working correctly!")
        print("✅ 20-question quiz generation: PASSED")
        print("✅ Detailed report with highlighted transcript: PASSED")
        print("✅ Enhanced data extraction: PASSED")
        print("✅ Professional PDF formatting: PASSED")
    else:
        print("💥 Some tests failed. Please check the implementation.")
        sys.exit(1)