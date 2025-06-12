from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
# from flask import Flask, request, jsonify, send_from_directory,


# Import your existing modules
from pdfextractor import extract_pdf_text
from keywords_extractor import extract_keywords_from_pdf_json
import question_generator 
from jsonfilter import filter_keywords_from_file

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, DATA_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'session_id': session_id,
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process/<session_id>', methods=['POST'])
def process_pdf(session_id):
    try:
        # Find the uploaded file
        upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(session_id)]
        if not upload_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = os.path.join(UPLOAD_FOLDER, upload_files[0])
        filename = upload_files[0].split('_', 1)[1]  # Remove session_id prefix
        pdf_filename = os.path.splitext(filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Extract text from PDF
        print("Extracting text...")
        extracted_text = extract_pdf_text(file_path, "structured")
        
        # Save extracted text
        text_output_filename = os.path.join(DATA_FOLDER, f"text_extraction_result_{session_id}_{timestamp}.json")
        with open(text_output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(extracted_text, json_file, indent=2, ensure_ascii=False, default=str)
        
        # Step 2: Extract keywords
        print("Extracting keywords...")
        keywords_json_file_path = os.path.join(DATA_FOLDER, f"keyword_extraction_result_{session_id}_{timestamp}.json")
        extracted_keywords = extract_keywords_from_pdf_json(text_output_filename, 5, keywords_json_file_path)
        
        if not extracted_keywords['success']:
            return jsonify({'error': 'Failed to extract keywords'}), 500
        
        # Step 3: Filter keywords
        print("Filtering keywords...")
        keywords_list = filter_keywords_from_file(keywords_json_file_path, 0.35)
        
        # Step 4: Generate MCQs
        print("Generating MCQs...")
        mcqs = question_generator.generate_mcqs(keywords_list, 10)
        
        if not mcqs:
            return jsonify({'error': 'Failed to generate MCQs'}), 500
        
        # Save MCQs
        mcqs_filename = os.path.join(DATA_FOLDER, f"mcqs_{session_id}_{timestamp}.json")
        question_generator.save_mcqs_to_file(mcqs, mcqs_filename)
        
        # Return MCQs to frontend
        return jsonify({
            'message': 'Processing completed successfully',
            'session_id': session_id,
            'mcqs': mcqs,
            'total_questions': len(mcqs)
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/submit-answers', methods=['POST'])
def submit_answers():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_answers = data.get('answers')
        
        if not session_id or not user_answers:
            return jsonify({'error': 'Missing session_id or answers'}), 400
        
        # Load the original MCQs to compare answers
        mcq_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith(f"mcqs_{session_id}")]
        if not mcq_files:
            return jsonify({'error': 'MCQ file not found'}), 404
        
        mcq_file_path = os.path.join(DATA_FOLDER, mcq_files[0])
        with open(mcq_file_path, 'r', encoding='utf-8') as f:
            mcqs = json.load(f)
        
        # Calculate results
        total_questions = len(mcqs)
        correct_answers = 0
        results = []
        
        for i, mcq in enumerate(mcqs):
            user_answer = user_answers.get(str(i))
            is_correct = user_answer == mcq['correct_answer']
            if is_correct:
                correct_answers += 1
            
            results.append({
                'question_index': i,
                'question': mcq['question'],
                'user_answer': user_answer,
                'correct_answer': mcq['correct_answer'],
                'is_correct': is_correct,
                'explanation': mcq['explanation']
            })
        
        # Calculate score
        score = (correct_answers / total_questions) * 100
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data = {
            'session_id': session_id,
            'timestamp': timestamp,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'score': score,
            'results': results
        }
        
        results_filename = os.path.join(RESULTS_FOLDER, f"results_{session_id}_{timestamp}.json")
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'message': 'Answers submitted successfully',
            'score': score,
            'correct_answers': correct_answers,
            'total_questions': total_questions,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to submit answers: {str(e)}'}), 500

@app.route('/api/results/<session_id>', methods=['GET'])
def get_results(session_id):
    try:
        # Find result files for this session
        result_files = [f for f in os.listdir(RESULTS_FOLDER) if f.startswith(f"results_{session_id}")]
        if not result_files:
            return jsonify({'error': 'Results not found'}), 404
        
        # Get the most recent result file
        result_files.sort(reverse=True)
        result_file_path = os.path.join(RESULTS_FOLDER, result_files[0])
        
        with open(result_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # if a static file exists, serve it
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # otherwise serve index.html
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
