import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()  # ← this reads your .env into os.environ


# Best practice: Store your API key as an environment variable
# For testing, you can replace os.environ.get("OPENROUTER_API_KEY") directly,
# but avoid hardcoding keys in production code.
router_key = os.getenv("OPENROUTER_API_KEY")
if not router_key:
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment")

client = openai.OpenAI(
    api_key=router_key,
    base_url="https://openrouter.ai/api/v1"
)

def generate_mcqs(keywords, num_questions=10):
    """
    Generate multiple choice questions based on keywords
    
    Args:
        keywords (list): List of keywords to generate MCQs from
        num_questions (int): Number of MCQs to generate (default: 10)
    
    Returns:
        list: List of dictionaries containing MCQs with answers
    """
    
    # Convert keywords list to string
    keywords_str = ", ".join(keywords)
    
    # Create a detailed prompt for MCQ generation
    prompt = f"""
    Generate exactly {num_questions} multiple choice questions based on these keywords: {keywords_str}
    
    For each question:
    1. Create a clear, educational question related to the concepts of the given keywords
    2. Provide 4 answer options (A, B, C, D)
    3. One option should be clearly correct
    4. The other 3 should be plausible but incorrect
    5. Indicate which option is correct
    
    Format your response as a JSON array where each question is an object with this structure:
    {{
        "question": "Question text here?",
        "options": {{
            "A": "Option A text",
            "B": "Option B text", 
            "C": "Option C text",
            "D": "Option D text"
        }},
        "correct_answer": "A",
        "explanation": "Brief explanation of why this answer is correct"
    }}
    
    Make sure the questions cover different aspects and difficulty levels of the keywords provided.
    Return only the JSON array, no additional text.
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528-qwen3-8b:free",
            messages=[
                {"role": "system", "content": "You are an expert educator who creates high-quality multiple choice questions. Always respond with valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=20000
        )
        
        # Parse the JSON response
        mcqs_text = response.choices[0].message.content.strip()
        
        # Remove any markdown formatting if present
        if mcqs_text.startswith("```json"):
            mcqs_text = mcqs_text[7:-3]
        elif mcqs_text.startswith("```"):
            mcqs_text = mcqs_text[3:-3]
            
        mcqs = json.loads(mcqs_text)
        return mcqs
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return None
    except openai.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def display_mcqs(mcqs):
    """
    Display MCQs in a formatted way
    
    Args:
        mcqs (list): List of MCQ dictionaries
    """
    if not mcqs:
        print("No MCQs to display.")
        return
    
    print("=" * 60)
    print("GENERATED MULTIPLE CHOICE QUESTIONS")
    print("=" * 60)
    
    for i, mcq in enumerate(mcqs, 1):
        print(f"\nQuestion {i}: {mcq['question']}")
        print("-" * 40)
        
        for option, text in mcq['options'].items():
            marker = "✓" if option == mcq['correct_answer'] else " "
            print(f"{marker} {option}. {text}")
        
        print(f"\nCorrect Answer: {mcq['correct_answer']}")
        print(f"Explanation: {mcq['explanation']}")
        print("-" * 60)

def save_mcqs_to_file(mcqs, filename="generated_mcqs.json"):
    """
    Save MCQs to a JSON file
    
    Args:
        mcqs (list): List of MCQ dictionaries
        filename (str): Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(mcqs, f, indent=2, ensure_ascii=False)
        print(f"\nMCQs saved to {filename}")
    except Exception as e:
        print(f"Error saving MCQs to file: {e}")

# Example usage
if __name__ == "__main__":
    # Define your keywords here
    keywords = ['iaas service', 'infrastructure platform', 'software applications', 'components', 'frameworks', 'storage', 'servers', 'control focus', 'solutions', 'resources', 'configurations', 'automatic updates', 'automatic', 'environments', 'data', 'platform', 'updates', 'high infrastructure', 'configuration', 'management', 'virtual machines', 'virtual', 'machines', 'networks', 'beanstalk', 'workspace', 'google compute', 'google', 'compute', 'virtualized platform', 'virtualized', 'engine', 'compute resources', 'host', 'service', 'web apps', 'apps', 'applications communication tools', 'machines applications', 'communication tools', 'applications communication', 'model', 'administrators', 'devops', 'architects', 'development cycles', 'accessible', 'businesses', 'disadvantages', 'expertise', 'infrastructure customization', 'customization', 'risks', 'dependency', 'oauth', 'application https', 'xen', 'django', 'jenkins', 'tools jenkins', 'complete', 'hardware', 'testing', 'hosting complex', 'complex']
    
    print("Generating MCQs for keywords:", ", ".join(keywords))
    print("Please wait...")
    
    # Generate MCQs
    mcqs = generate_mcqs(keywords, num_questions=10)
    
    if mcqs:
        # Display the generated MCQs
        display_mcqs(mcqs)
        
        # Save to file
        save_mcqs_to_file(mcqs)
        
        print(f"\nSuccessfully generated {len(mcqs)} MCQs!")
    else:
        print("Failed to generate MCQs. Please check your API key and try again.")
