import json

def filter_keywords_by_threshold(json_data, threshold):
    """
    Filters keywords from the JSON extraction result based on a score threshold.
    
    Args:
        json_data (dict or str): The JSON data (as dict) or JSON string containing keyword extraction results
        threshold (float): The minimum score threshold for keywords to be included
    
    Returns:
        list: List of keywords (strings) that meet or exceed the threshold, in original order
    """
    # Handle string input (JSON string)
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract keywords that meet the threshold
    filtered_keywords = []
    
    # Check if the JSON has the expected structure
    if 'keywords' in data and isinstance(data['keywords'], list):
        for keyword_obj in data['keywords']:
            if 'keyword' in keyword_obj and 'score' in keyword_obj:
                if keyword_obj['score'] >= threshold:
                    filtered_keywords.append(keyword_obj['keyword'])
    else:
        raise ValueError("Invalid JSON structure: 'keywords' array not found or invalid")
    
    return filtered_keywords


def filter_keywords_from_file(file_path, threshold):
    """
    Reads JSON from a file and filters keywords by threshold.
    
    Args:
        file_path (str): Path to the JSON file
        threshold (float): The minimum score threshold for keywords
    
    Returns:
        list: List of filtered keywords
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    return filter_keywords_by_threshold(json_data, threshold)


# Example usage:
if __name__ == "__main__":
    # Get file path from user input
    json_file_path = "keyword_extraction_results.json"
    threshold = 0.4
    
    
    try:
        # Filter keywords from file
        result = filter_keywords_from_file(json_file_path, threshold)
        print(result)
        
        print(f"\nKeywords with score >= {threshold} from '{json_file_path}':")
        print(f"Found {len(result)} keywords:")
        print("-" * 50)
        for i, keyword in enumerate(result, 1):
            print(f"{i:2d}. {keyword}")
            
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")