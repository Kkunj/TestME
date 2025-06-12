from pdfextractor import extract_pdf_text
from keywords_extractor import extract_keywords_from_pdf_json
import question_generator 
from jsonfilter import filter_keywords_from_file
import json
from datetime import datetime
import os

sample_path = "pdf_test.pdf"

print("extracting text...")

extracted_text = extract_pdf_text(sample_path,"structured" )

print(type(extracted_text))
print("text extracted.")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_filename = os.path.splitext(os.path.basename(sample_path))[0]
output_filename = f"data/text_extraction_result_{pdf_filename}_{timestamp}.json"

try:
    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(extracted_text, json_file, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Extraction result saved to: {output_filename}")
except Exception as e:
    print(f"❌ Failed to save text extraction JSON file: {str(e)}")

print("extracting keywords")
keywords_json_file_path = f"data/keyword_extraction_result_{pdf_filename}_{timestamp}.json"
extracted_keywords = extract_keywords_from_pdf_json(output_filename, 5, keywords_json_file_path)

if extracted_keywords['success']:
    print("Keywords extracted successfully")
else:
    print("error extracting keywords")



print("filetering keywords")
keywords_list = filter_keywords_from_file(keywords_json_file_path, 0.35)
print("keywords filtered")
mcqs = question_generator.generate_mcqs(keywords_list, 10)
# mcqs = uqegenerate_mcqs(keywords, num_questions=10)

if mcqs:
        # Display the generated MCQs
    question_generator.display_mcqs(mcqs)
        
        # Save to file
    mcqs_filename = f"data/mcqs_{pdf_filename}_{timestamp}.json"
    question_generator.save_mcqs_to_file(mcqs, mcqs_filename)
        # questionsave_mcqs_to_file(mcqs)
        
    print(f"\nSuccessfully generated {len(mcqs)} MCQs!")
else:
    print("Failed to generate MCQs. Please check your API key and try again.")