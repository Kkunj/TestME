# keyword_extractor.py
"""
Topic & Keyword Extraction System for Educational Content
Processes structured JSON output from PDF extractor and extracts high-value keywords
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import string

# Import required libraries for keyword extraction
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
except ImportError:
    print("❌ spaCy not installed. Install with: pip install spacy")
    print("❌ Also run: python -m spacy download en_core_web_sm")
    spacy = None
    STOP_WORDS = set()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("❌ scikit-learn not installed. Install with: pip install scikit-learn")
    TfidfVectorizer = None

try:
    import yake
except ImportError:
    print("❌ YAKE not installed. Install with: pip install yake")
    yake = None

try:
    from rake_nltk import Rake
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
except ImportError:
    print("❌ RAKE-NLTK not installed. Install with: pip install rake-nltk")
    Rake = None


class KeywordExtractor:
    """
    Comprehensive keyword extraction system for educational content.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize the keyword extractor with required models and settings."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy model
        self.nlp = self._load_spacy_model()
        
        # Initialize RAKE extractor
        self.rake = self._initialize_rake()
        
        # Initialize YAKE extractor
        self.yake_extractor = self._initialize_yake()
        
        # Custom stop words for educational content
        self.custom_stopwords = self._get_custom_stopwords()
        
    def setup_logging(self, level: str) -> None:
        """Configure logging for the extractor."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling."""
        if spacy is None:
            self.logger.warning("spaCy not available. Some features will be limited.")
            return None
        
        try:
            nlp = spacy.load("en_core_web_sm")
            self.logger.info("✅ spaCy model 'en_core_web_sm' loaded successfully")
            return nlp
        except OSError:
            try:
                nlp = spacy.load("en_core_web_md")
                self.logger.info("✅ spaCy model 'en_core_web_md' loaded successfully")
                return nlp
            except OSError:
                self.logger.warning("❌ No spaCy model found. Install with: python -m spacy download en_core_web_sm")
                return None
    
    def _initialize_rake(self):
        """Initialize RAKE keyword extractor."""
        if Rake is None:
            self.logger.warning("RAKE not available. Install with: pip install rake-nltk")
            return None
        
        try:
            rake = Rake()
            self.logger.info("✅ RAKE extractor initialized")
            return rake
        except Exception as e:
            self.logger.warning(f"Failed to initialize RAKE: {str(e)}")
            return None
    
    def _initialize_yake(self):
        """Initialize YAKE keyword extractor."""
        if yake is None:
            self.logger.warning("YAKE not available. Install with: pip install yake")
            return None
        
        try:
            # Configure YAKE parameters for educational content
            yake_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Maximum number of words in keyphrase
                dedupLim=0.9,  # Deduplication threshold
                top=10,  # Number of keywords to extract
                features=None
            )
            self.logger.info("✅ YAKE extractor initialized")
            return yake_extractor
        except Exception as e:
            self.logger.warning(f"Failed to initialize YAKE: {str(e)}")
            return None
    
    def _get_custom_stopwords(self) -> set:
        """Get custom stopwords for educational content."""
        educational_stopwords = {
            'chapter', 'section', 'page', 'figure', 'table', 'example', 'exercise',
            'question', 'answer', 'note', 'summary', 'conclusion', 'introduction',
            'overview', 'review', 'study', 'learning', 'objective', 'goal',
            'lecture', 'slide', 'presentation', 'course', 'unit', 'lesson'
        }
        
        # Combine with spaCy stopwords if available
        if STOP_WORDS:
            return STOP_WORDS.union(educational_stopwords)
        else:
            # Basic English stopwords if spaCy not available
            basic_stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'were', 'will', 'with', 'the', 'this', 'these', 'those'
            }
            return basic_stopwords.union(educational_stopwords)
    
    def load_and_inspect_json(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load and inspect the JSON structure from PDF extraction.
        
        Args:
            json_file_path (str): Path to the JSON file from PDF extraction
            
        Returns:
            Dict containing loaded JSON data and structure analysis
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if not data.get('success', False):
                return {
                    'success': False,
                    'error': f"PDF extraction was not successful: {data.get('error', 'Unknown error')}"
                }
            
            # Inspect structure
            extracted_text = data.get('extracted_text', {})
            pages = extracted_text.get('pages', [])
            
            structure_info = {
                'total_pages': len(pages),
                'pages_with_structure': 0,
                'total_headings': 0,
                'total_paragraphs': 0,
                'total_bullet_points': 0,
                'extraction_mode': data.get('extraction_info', {}).get('extraction_mode', 'unknown')
            }
            
            # Count structural elements
            for page in pages:
                if 'structure' in page:
                    structure_info['pages_with_structure'] += 1
                    structure = page['structure']
                    structure_info['total_headings'] += len(structure.get('headings', []))
                    structure_info['total_paragraphs'] += len(structure.get('paragraphs', []))
                    structure_info['total_bullet_points'] += len(structure.get('bullet_points', []))
            
            return {
                'success': True,
                'data': data,
                'structure_info': structure_info
            }
            
        except FileNotFoundError:
            return {'success': False, 'error': f"JSON file not found: {json_file_path}"}
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f"Invalid JSON format: {str(e)}"}
        except Exception as e:
            return {'success': False, 'error': f"Error loading JSON: {str(e)}"}
    
    def define_logical_units(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Define logical units from the JSON structure.
        
        Args:
            json_data (Dict): Loaded JSON data from PDF extraction
            
        Returns:
            List of logical units with preserved ordering
        """
        extracted_text = json_data.get('extracted_text', {})
        pages = extracted_text.get('pages', [])
        
        logical_units = []
        current_section = None
        section_index = 0
        
        for page in pages:
            page_number = page.get('page_number', 0)
            
            # Check if page has structured data
            if 'structure' in page:
                structure = page['structure']
                headings = structure.get('headings', [])
                paragraphs = structure.get('paragraphs', [])
                bullet_points = structure.get('bullet_points', [])
                
                # Process headings to create new sections
                for heading in headings:
                    # Save previous section if exists
                    if current_section:
                        logical_units.append(current_section)
                    
                    # Start new section
                    section_index += 1
                    current_section = {
                        'section_index': section_index,
                        'section_title': heading['text'],
                        'page_number': page_number,
                        'line_number': heading.get('line_number', 0),
                        'text_blocks': [heading['text']],
                        'content_type': 'structured'
                    }
                
                # Add paragraphs to current section
                for paragraph in paragraphs:
                    if current_section:
                        current_section['text_blocks'].append(paragraph['text'])
                    else:
                        # Create section if no heading found
                        section_index += 1
                        current_section = {
                            'section_index': section_index,
                            'section_title': f"Page {page_number} Content",
                            'page_number': page_number,
                            'line_number': paragraph.get('line_number', 0),
                            'text_blocks': [paragraph['text']],
                            'content_type': 'structured'
                        }
                
                # Add bullet points to current section
                for bullet in bullet_points:
                    if current_section:
                        current_section['text_blocks'].append(bullet['text'])
                    else:
                        # Create section if no heading found
                        section_index += 1
                        current_section = {
                            'section_index': section_index,
                            'section_title': f"Page {page_number} Content",
                            'page_number': page_number,
                            'line_number': bullet.get('line_number', 0),
                            'text_blocks': [bullet['text']],
                            'content_type': 'structured'
                        }
            
            else:
                # Handle pages without structure (simple extraction)
                page_text = page.get('text', '')
                if page_text.strip():
                    section_index += 1
                    logical_units.append({
                        'section_index': section_index,
                        'section_title': f"Page {page_number}",
                        'page_number': page_number,
                        'line_number': 1,
                        'text_blocks': [page_text],
                        'content_type': 'simple'
                    })
        
        # Add the last section
        if current_section:
            logical_units.append(current_section)
        
        return logical_units
    
    def preprocess_text_blocks(self, logical_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pre-process text blocks in logical units.
        
        Args:
            logical_units (List): List of logical units to process
            
        Returns:
            List of processed logical units
        """
        processed_units = []
        
        for unit in logical_units:
            # Combine all text blocks into a single text
            combined_text = ' '.join(unit['text_blocks'])
            
            # Clean the text
            cleaned_text = self._clean_text(combined_text)
            
            # Create processed unit
            processed_unit = unit.copy()
            processed_unit['original_text'] = combined_text
            processed_unit['cleaned_text'] = cleaned_text
            processed_unit['word_count'] = len(cleaned_text.split())
            
            # Only keep units with meaningful content
            if len(cleaned_text.split()) >= 3:  # At least 3 words
                processed_units.append(processed_unit)
        
        return processed_units
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove OCR artifacts and hyphenation at line breaks
        text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenation
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with space
        
        # Remove special characters but keep punctuation that matters
        text = re.sub(r'[^\w\s\-\.\,\:\;\!\?]', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\-]{2,}', '-', text)
        
        # Normalize case (keep original case for proper nouns detection)
        text = text.strip()
        
        return text
    
    def extract_keywords_rake(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Extract keywords using RAKE algorithm."""
        if self.rake is None or not text.strip():
            return []
        
        try:
            self.rake.extract_keywords_from_text(text)
            keywords_with_scores = self.rake.get_ranked_phrases_with_scores()
            
            # Return top N keywords
            return keywords_with_scores[:top_n]
        except Exception as e:
            self.logger.warning(f"RAKE extraction failed: {str(e)}")
            return []
    
    def extract_keywords_yake(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE algorithm."""
        if self.yake_extractor is None or not text.strip():
            return []
        
        try:
            keywords = self.yake_extractor.extract_keywords(text)
            # YAKE returns (keyword, score) where lower score is better
            # Return top N keywords
            return keywords[:top_n]
        except Exception as e:
            self.logger.warning(f"YAKE extraction failed: {str(e)}")
            return []
    
    def extract_keywords_spacy_tfidf(self, processed_units: List[Dict[str, Any]], top_n: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """Extract keywords using spaCy + TF-IDF."""
        if self.nlp is None or TfidfVectorizer is None:
            return {}
        
        # Prepare documents for TF-IDF
        documents = []
        unit_mapping = {}
        
        for i, unit in enumerate(processed_units):
            doc = self.nlp(unit['cleaned_text'])
            
            # Extract noun chunks and proper nouns
            candidates = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Max 3 words
                    candidates.append(chunk.text.lower().strip())
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    candidates.append(ent.text.lower().strip())
            
            # Filter out stopwords and short words
            filtered_candidates = []
            for candidate in candidates:
                words = candidate.split()
                if (len(words) >= 1 and 
                    not any(word in self.custom_stopwords for word in words) and
                    len(candidate) > 2):
                    filtered_candidates.append(candidate)
            
            document_text = ' '.join(filtered_candidates)
            documents.append(document_text)
            unit_mapping[i] = unit['section_index']
        
        if not documents or not any(doc.strip() for doc in documents):
            return {}
        
        try:
            # Apply TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            results = {}
            
            for i, unit in enumerate(processed_units):
                if i < len(tfidf_matrix.toarray()):
                    # Get TF-IDF scores for this document
                    scores = tfidf_matrix.toarray()[i]
                    
                    # Get top keywords with scores
                    keyword_scores = []
                    for j, score in enumerate(scores):
                        if score > 0:
                            keyword_scores.append((feature_names[j], score))
                    
                    # Sort by score and get top N
                    keyword_scores.sort(key=lambda x: x[1], reverse=True)
                    results[unit['section_index']] = keyword_scores[:top_n]
            
            return results
            
        except Exception as e:
            self.logger.warning(f"spaCy + TF-IDF extraction failed: {str(e)}")
            return {}
    
    def extract_keywords_comprehensive(self, json_file_path: str, top_keywords_per_section: int = 5) -> Dict[str, Any]:
        """
        Comprehensive keyword extraction using all available methods.
        
        Args:
            json_file_path (str): Path to JSON file from PDF extraction
            top_keywords_per_section (int): Number of top keywords per section
            
        Returns:
            Dict containing extracted keywords and metadata
        """
        # Step 1: Load and inspect JSON
        self.logger.info("Step 1: Loading and inspecting JSON structure...")
        load_result = self.load_and_inspect_json(json_file_path)
        
        if not load_result['success']:
            return {
                'success': False,
                'error': load_result['error'],
                'timestamp': datetime.now().isoformat()
            }
        
        json_data = load_result['data']
        structure_info = load_result['structure_info']
        
        # Step 2: Define logical units
        self.logger.info("Step 2: Defining logical units...")
        logical_units = self.define_logical_units(json_data)
        
        # Step 3: Pre-process text blocks
        self.logger.info("Step 3: Pre-processing text blocks...")
        processed_units = self.preprocess_text_blocks(logical_units)
        
        # Step 4: Extract keywords using multiple techniques
        self.logger.info("Step 4: Extracting keywords using multiple techniques...")
        
        all_keywords = []
        section_keywords = {}
        
        # Extract using spaCy + TF-IDF (global analysis)
        spacy_tfidf_results = self.extract_keywords_spacy_tfidf(processed_units, top_keywords_per_section)
        
        for unit in processed_units:
            section_index = unit['section_index']
            section_title = unit['section_title']
            page_number = unit['page_number']
            cleaned_text = unit['cleaned_text']
            
            # Extract keywords using RAKE
            rake_keywords = self.extract_keywords_rake(cleaned_text, top_keywords_per_section)
            
            # Extract keywords using YAKE
            yake_keywords = self.extract_keywords_yake(cleaned_text, top_keywords_per_section)
            
            # Get spaCy + TF-IDF keywords for this section
            spacy_keywords = spacy_tfidf_results.get(section_index, [])
            
            # Combine and rank keywords
            combined_keywords = self._combine_keyword_results(
                rake_keywords, yake_keywords, spacy_keywords, top_keywords_per_section
            )
            
            # Store section keywords
            section_keywords[section_index] = {
                'section_title': section_title,
                'page_number': page_number,
                'keywords': combined_keywords,
                'rake_keywords': rake_keywords,
                'yake_keywords': yake_keywords,
                'spacy_tfidf_keywords': spacy_keywords,
                'text_snippet': cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
                'word_count': unit['word_count']
            }
            
            # Add to master list with sequential ordering
            for rank, (keyword, score) in enumerate(combined_keywords, 1):
                keyword_entry = {
                    'keyword': keyword,
                    'section_index': section_index,
                    'page_number': page_number,
                    'original_text_snippet': cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
                    'rank': rank,
                    'score': score,
                    'section_title': section_title,
                    'extraction_methods': self._get_keyword_sources(keyword, rake_keywords, yake_keywords, spacy_keywords)
                }
                all_keywords.append(keyword_entry)
        
        # Step 5: Sequential ordering and deduplication
        self.logger.info("Step 5: Applying sequential ordering and deduplication...")
        deduplicated_keywords = self._deduplicate_keywords(all_keywords)
        
        # Prepare final result
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'source_file': json_file_path,
            'extraction_summary': {
                'total_sections': len(processed_units),
                'total_keywords_extracted': len(deduplicated_keywords),
                'keywords_per_section': top_keywords_per_section,
                'extraction_methods': ['RAKE', 'YAKE', 'spaCy + TF-IDF'],
                'structure_info': structure_info
            },
            'keywords': deduplicated_keywords,
            'sections': section_keywords,
            'processing_info': {
                'logical_units_created': len(logical_units),
                'processed_units': len(processed_units),
                'total_text_blocks': sum(len(unit['text_blocks']) for unit in logical_units)
            }
        }
        
        return result
    
    def _combine_keyword_results(
        self, 
        rake_keywords: List[Tuple[str, float]], 
        yake_keywords: List[Tuple[str, float]], 
        spacy_keywords: List[Tuple[str, float]], 
        top_n: int
    ) -> List[Tuple[str, float]]:
        """Combine results from different keyword extraction methods."""
        keyword_scores = defaultdict(list)
        
        # Add RAKE scores (higher is better)
        for keyword, score in rake_keywords:
            if isinstance(keyword, str):
                keyword_scores[keyword.lower().strip()].append(('rake', score))
        
        # Add YAKE scores (lower is better, so invert)
        for keyword, score in yake_keywords:
            if isinstance(keyword, str):
                # Invert YAKE score (lower is better → higher is better)
                inverted_score = 1.0 / (1.0 + score) if score > 0 else 1.0
                keyword_scores[keyword.lower().strip()].append(('yake', inverted_score))
        
        # Add spaCy + TF-IDF scores (higher is better)
        for keyword, score in spacy_keywords:
            if isinstance(keyword, str):
                keyword_scores[keyword.lower().strip()].append(('spacy_tfidf', score))
        
        # Calculate combined scores
        final_keywords = []
        for keyword, scores in keyword_scores.items():
            if len(keyword.strip()) > 2:  # Filter very short keywords
                # Weight different methods
                total_score = 0
                method_count = len(scores)
                
                for method, score in scores:
                    if method == 'rake':
                        total_score += score * 0.4  # RAKE weight
                    elif method == 'yake':
                        total_score += score * 0.3  # YAKE weight
                    elif method == 'spacy_tfidf':
                        total_score += score * 0.3  # spaCy + TF-IDF weight
                
                # Bonus for keywords found by multiple methods
                if method_count > 1:
                    total_score *= 1.2
                
                final_keywords.append((keyword, total_score))
        
        # Sort by score and return top N
        final_keywords.sort(key=lambda x: x[1], reverse=True)
        return final_keywords[:top_n]
    
    def _get_keyword_sources(
        self, 
        keyword: str, 
        rake_keywords: List[Tuple[str, float]], 
        yake_keywords: List[Tuple[str, float]], 
        spacy_keywords: List[Tuple[str, float]]
    ) -> List[str]:
        """Get the sources/methods that identified a keyword."""
        sources = []
        keyword_lower = keyword.lower().strip()
        
        if any(isinstance(kw, str) and kw.lower().strip() == keyword_lower for kw, _ in rake_keywords):
            sources.append('RAKE')
        if any(isinstance(kw, str) and kw.lower().strip() == keyword_lower for kw, _ in yake_keywords):
            sources.append('YAKE')
        if any(isinstance(kw, str) and kw.lower().strip() == keyword_lower for kw, _ in spacy_keywords):
            sources.append('spaCy_TF-IDF')
        
        return sources
    
    def _deduplicate_keywords(self, all_keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate keywords while preserving first occurrence."""
        seen_keywords = set()
        deduplicated = []
        
        for keyword_entry in all_keywords:
            keyword_normalized = keyword_entry['keyword'].lower().strip()
            
            if keyword_normalized not in seen_keywords:
                seen_keywords.add(keyword_normalized)
                deduplicated.append(keyword_entry)
        
        return deduplicated


# Main function for external access
def extract_keywords_from_pdf_json(
    json_file_path: str, 
    top_keywords_per_section: int = 5,
    output_file: Optional[str] = None,
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """
    Main function to extract keywords from PDF extraction JSON - designed for external access.
    
    Args:
        json_file_path (str): Path to JSON file from PDF extraction
        top_keywords_per_section (int): Number of top keywords per section
        output_file (str): Optional output file path for results
        log_level (str): Logging level
        
    Returns:
        Dict containing keyword extraction results
    """
    extractor = KeywordExtractor(log_level=log_level)
    results = extractor.extract_keywords_comprehensive(json_file_path, top_keywords_per_section)
    
    # Save to file if requested
    if output_file and results['success']:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            results['output_file'] = output_file
        except Exception as e:
            results['save_error'] = str(e)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    sample_json_path = "extraction_result_pdf_test_20250610_225535.json"  # Replace with actual path
    
    # Extract keywords
    result = extract_keywords_from_pdf_json(
        sample_json_path, 
        top_keywords_per_section=5,
        output_file="keyword_extraction_results.json"
    )
    
    if result['success']:
        print(f"✅ Successfully extracted keywords from {result['extraction_summary']['total_sections']} sections")
        print(f"📊 Total keywords extracted: {result['extraction_summary']['total_keywords_extracted']}")
        print(f"🎯 Keywords per section: {result['extraction_summary']['keywords_per_section']}")
        print(f"🔧 Methods used: {', '.join(result['extraction_summary']['extraction_methods'])}")
        
        # Show sample keywords
        print(f"\n🔑 Sample keywords from first few sections:")
        for i, keyword_entry in enumerate(result['keywords'][:10]):  # First 10 keywords
            print(f"   {i+1}. '{keyword_entry['keyword']}' (Section {keyword_entry['section_index']}, "
                  f"Page {keyword_entry['page_number']}, Score: {keyword_entry['score']:.3f})")
        
        if 'output_file' in result:
            print(f"\n💾 Results saved to: {result['output_file']}")
        
    else:
        print(f"❌ Keyword extraction failed: {result['error']}")