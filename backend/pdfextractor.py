# pdf_extractor.py
"""
PDF Text Extraction System for Educational Content Processing
Designed for extracting text from lecture slides and textbooks
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pypdf
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractor:
    """
    A comprehensive PDF text extraction class optimized for educational content.
    Handles various PDF layouts and provides multiple extraction modes.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the PDF extractor with logging configuration.
        
        Args:
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self, level: str) -> None:
        """Configure logging for the extractor."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def extract_text_from_pdf(
        self, 
        file_path: str, 
        extraction_mode: str = "simple",
        page_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, any]:
        """
        Extract text from PDF file with various extraction modes.
        
        Args:
            file_path (str): Path to the PDF file
            extraction_mode (str): Extraction mode ('simple', 'layout_aware', 'structured')
            page_range (tuple): Optional tuple (start_page, end_page) for partial extraction
            
        Returns:
            Dict containing extracted text, metadata, and extraction info
        """
        try:
            # Validate file path
            if not self._validate_pdf_file(file_path):
                return self._create_error_response("Invalid PDF file path or file doesn't exist")
            
            # Read PDF
            reader = PdfReader(file_path)
            
            # Get PDF metadata
            metadata = self._extract_metadata(reader)
            
            # Determine page range
            total_pages = len(reader.pages)
            start_page, end_page = self._determine_page_range(page_range, total_pages)
            
            # Extract text based on mode
            if extraction_mode == "simple":
                extracted_text = self._simple_extraction(reader, start_page, end_page)
            elif extraction_mode == "layout_aware":
                extracted_text = self._layout_aware_extraction(reader, start_page, end_page)
            elif extraction_mode == "structured":
                extracted_text = self._structured_extraction(reader, start_page, end_page)
            else:
                return self._create_error_response(f"Unknown extraction mode: {extraction_mode}")
            
            return self._create_success_response(
                extracted_text, metadata, total_pages, extraction_mode, start_page, end_page
            )
            
        except PdfReadError as e:
            self.logger.error(f"PDF reading error: {str(e)}")
            return self._create_error_response(f"PDF reading error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during extraction: {str(e)}")
            return self._create_error_response(f"Extraction failed: {str(e)}")
    
    def _validate_pdf_file(self, file_path: str) -> bool:
        """Validate if the file exists and is a PDF."""
        path = Path(file_path)
        return path.exists() and path.suffix.lower() == '.pdf'
    
    def _extract_metadata(self, reader: PdfReader) -> Dict[str, any]:
        """Extract metadata from PDF."""
        metadata = {}
        try:
            if reader.metadata:
                metadata = {
                    'title': reader.metadata.get('/Title', 'Unknown'),
                    'author': reader.metadata.get('/Author', 'Unknown'),
                    'subject': reader.metadata.get('/Subject', 'Unknown'),
                    'creator': reader.metadata.get('/Creator', 'Unknown'),
                    'producer': reader.metadata.get('/Producer', 'Unknown'),
                    'creation_date': str(reader.metadata.get('/CreationDate', 'Unknown')),
                    'modification_date': str(reader.metadata.get('/ModDate', 'Unknown'))
                }
        except Exception as e:
            self.logger.warning(f"Could not extract metadata: {str(e)}")
            metadata = {'error': 'Metadata extraction failed'}
        
        return metadata
    
    def _determine_page_range(self, page_range: Optional[Tuple[int, int]], total_pages: int) -> Tuple[int, int]:
        """Determine the actual page range to process."""
        if page_range is None:
            return 0, total_pages
        
        start_page = max(0, page_range[0])
        end_page = min(total_pages, page_range[1])
        
        return start_page, end_page
    
    def _simple_extraction(self, reader: PdfReader, start_page: int, end_page: int) -> Dict[str, any]:
        """Simple text extraction without layout preservation."""
        pages_text = []
        full_text = ""
        
        for page_num in range(start_page, end_page):
            try:
                page = reader.pages[page_num]
                page_text = page.extract_text()
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'word_count': len(page_text.split())
                })
                full_text += page_text + "\n\n"
                
            except Exception as e:
                self.logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': '',
                    'error': str(e)
                })
        
        return {
            'full_text': full_text.strip(),
            'pages': pages_text,
            'total_words': len(full_text.split())
        }
    
    def _layout_aware_extraction(self, reader: PdfReader, start_page: int, end_page: int) -> Dict[str, any]:
        """Layout-aware text extraction to preserve text positioning."""
        pages_text = []
        full_text = ""
        
        for page_num in range(start_page, end_page):
            try:
                page = reader.pages[page_num]
                
                # Use layout mode for better text positioning
                page_text = page.extract_text(extraction_mode="layout")
                
                # Clean up the extracted text
                cleaned_text = self._clean_extracted_text(page_text)
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'extraction_mode': 'layout_aware'
                })
                full_text += cleaned_text + "\n\n"
                
            except Exception as e:
                self.logger.warning(f"Error in layout-aware extraction from page {page_num + 1}: {str(e)}")
                # Fallback to simple extraction
                try:
                    page_text = page.extract_text()
                    pages_text.append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'word_count': len(page_text.split()),
                        'extraction_mode': 'fallback_simple',
                        'warning': str(e)
                    })
                    full_text += page_text + "\n\n"
                except:
                    pages_text.append({
                        'page_number': page_num + 1,
                        'text': '',
                        'error': str(e)
                    })
        
        return {
            'full_text': full_text.strip(),
            'pages': pages_text,
            'total_words': len(full_text.split())
        }
    
    def _structured_extraction(self, reader: PdfReader, start_page: int, end_page: int) -> Dict[str, any]:
        """Structured extraction with additional text analysis."""
        pages_text = []
        full_text = ""
        document_structure = {
            'headings': [],
            'paragraphs': [],
            'bullet_points': [],
            'numbered_lists': []
        }
        
        for page_num in range(start_page, end_page):
            try:
                page = reader.pages[page_num]
                page_text = page.extract_text(extraction_mode="layout")
                cleaned_text = self._clean_extracted_text(page_text)
                
                # Analyze text structure
                page_structure = self._analyze_text_structure(cleaned_text, page_num + 1)
                
                # Merge with document structure
                for key in document_structure:
                    document_structure[key].extend(page_structure[key])
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'structure': page_structure
                })
                full_text += cleaned_text + "\n\n"
                
            except Exception as e:
                self.logger.warning(f"Error in structured extraction from page {page_num + 1}: {str(e)}")
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': '',
                    'error': str(e)
                })
        
        return {
            'full_text': full_text.strip(),
            'pages': pages_text,
            'total_words': len(full_text.split()),
            'document_structure': document_structure
        }
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = ' '.join(line.split())
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _analyze_text_structure(self, text: str, page_num: int) -> Dict[str, List[Dict]]:
        """Analyze text structure to identify headings, lists, etc."""
        lines = text.split('\n')
        structure = {
            'headings': [],
            'paragraphs': [],
            'bullet_points': [],
            'numbered_lists': []
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Identify potential headings (short lines, all caps, or specific patterns)
            if self._is_heading(line):
                structure['headings'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1
                })
            
            # Identify bullet points
            elif self._is_bullet_point(line):
                structure['bullet_points'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1
                })
            
            # Identify numbered lists
            elif self._is_numbered_list(line):
                structure['numbered_lists'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1
                })
            
            # Regular paragraphs
            else:
                structure['paragraphs'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1,
                    'word_count': len(line.split())
                })
        
        return structure
    
    def _is_heading(self, line: str) -> bool:
        """Determine if a line is likely a heading."""
        # Simple heuristics for heading detection
        return (
            len(line) < 100 and  # Short lines
            (line.isupper() or  # All uppercase
             len(line.split()) <= 8 or  # Few words
             line.endswith(':'))  # Ends with colon
        )
    
    def _is_bullet_point(self, line: str) -> bool:
        """Determine if a line is a bullet point."""
        return line.startswith(('•', '◦', '▪', '▫', '-', '*'))
    
    def _is_numbered_list(self, line: str) -> bool:
        """Determine if a line is part of a numbered list."""
        import re
        return bool(re.match(r'^\d+\.', line.strip()))
    
    def _create_success_response(
        self, 
        extracted_text: Dict, 
        metadata: Dict, 
        total_pages: int, 
        extraction_mode: str,
        start_page: int,
        end_page: int
    ) -> Dict[str, any]:
        """Create a successful extraction response."""
        return {
            'success': True,
            'extracted_text': extracted_text,
            'metadata': metadata,
            'extraction_info': {
                'total_pages': total_pages,
                'pages_processed': end_page - start_page,
                'extraction_mode': extraction_mode,
                'page_range': (start_page + 1, end_page)  # Convert to 1-based indexing
            },
            'error': None
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, any]:
        """Create an error response."""
        return {
            'success': False,
            'extracted_text': None,
            'metadata': None,
            'extraction_info': None,
            'error': error_message
        }
    
    def extract_text_from_multiple_pdfs(
        self, 
        file_paths: List[str], 
        extraction_mode: str = "simple"
    ) -> Dict[str, Dict[str, any]]:
        """
        Extract text from multiple PDF files.
        
        Args:
            file_paths (List[str]): List of PDF file paths
            extraction_mode (str): Extraction mode to use for all files
            
        Returns:
            Dict with file paths as keys and extraction results as values
        """
        results = {}
        
        for file_path in file_paths:
            self.logger.info(f"Processing: {file_path}")
            results[file_path] = self.extract_text_from_pdf(file_path, extraction_mode)
        
        return results
    
    def get_pdf_info(self, file_path: str) -> Dict[str, any]:
        """
        Get basic information about a PDF file without full text extraction.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict containing PDF information
        """
        try:
            if not self._validate_pdf_file(file_path):
                return {'error': 'Invalid PDF file path or file doesn\'t exist'}
            
            reader = PdfReader(file_path)
            metadata = self._extract_metadata(reader)
            
            return {
                'success': True,
                'file_path': file_path,
                'total_pages': len(reader.pages),
                'metadata': metadata,
                'file_size': os.path.getsize(file_path),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Main function for external access
def extract_pdf_text(
    file_path: str, 
    extraction_mode: str = "simple",
    page_range: Optional[Tuple[int, int]] = None,
    log_level: str = "INFO"
) -> Dict[str, any]:
    """
    Main function to extract text from PDF - designed for external access.
    
    Args:
        file_path (str): Path to the PDF file
        extraction_mode (str): 'simple', 'layout_aware', or 'structured'
        page_range (tuple): Optional (start_page, end_page) for partial extraction
        log_level (str): Logging level
        
    Returns:
        Dict containing extraction results
    """
    extractor = PDFExtractor(log_level=log_level)
    return extractor.extract_text_from_pdf(file_path, extraction_mode, page_range)


def extract_multiple_pdfs(
    file_paths: List[str], 
    extraction_mode: str = "simple",
    log_level: str = "INFO"
) -> Dict[str, Dict[str, any]]:
    """
    Extract text from multiple PDF files.
    
    Args:
        file_paths (List[str]): List of PDF file paths
        extraction_mode (str): Extraction mode to use
        log_level (str): Logging level
        
    Returns:
        Dict with results for each file
    """
    extractor = PDFExtractor(log_level=log_level)
    return extractor.extract_text_from_multiple_pdfs(file_paths, extraction_mode)


def get_pdf_info(file_path: str, log_level: str = "INFO") -> Dict[str, any]:
    """
    Get basic PDF information without full text extraction.
    
    Args:
        file_path (str): Path to the PDF file
        log_level (str): Logging level
        
    Returns:
        Dict containing PDF information
    """
    extractor = PDFExtractor(log_level=log_level)
    return extractor.get_pdf_info(file_path)


# Example usage and testing
# if __name__ == "__main__":
#     # Example usage
#     sample_pdf_path = "pdf_test.pdf"  # Replace with actual path
    
#     # Basic extraction
#     result = extract_pdf_text(sample_pdf_path, extraction_mode="structured")
    
#     if result['success']:
#         print(f"Successfully extracted text from {result['extraction_info']['pages_processed']} pages")
#         print(f"Total words: {result['extracted_text']['total_words']}")
#         print(f"Document title: {result['metadata'].get('title', 'Unknown')}")
#         print("\nFirst 500 characters of extracted text:")
#         print(result['extracted_text']['full_text'][:500] + "...")
#         print(result)
#     else:
#         print(f"Extraction failed: {result['error']}")

# Example usage and testing
if __name__ == "__main__":
    import json
    from datetime import datetime
    
    # Example usage
    sample_pdf_path = "pdf_test.pdf"  # Replace with actual path
    
    # Basic extraction
    result = extract_pdf_text(sample_pdf_path, extraction_mode="structured")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.splitext(os.path.basename(sample_pdf_path))[0]
    output_filename = f"extraction_result_{pdf_filename}_{timestamp}.json"
    
    # Save result to JSON file
    try:
        with open(output_filename, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Extraction result saved to: {output_filename}")
    except Exception as e:
        print(f"❌ Failed to save JSON file: {str(e)}")
    
    # Display summary
    if result['success']:
        print(f"Successfully extracted text from {result['extraction_info']['pages_processed']} pages")
        print(f"Total words: {result['extracted_text']['total_words']}")
        print(f"Document title: {result['metadata'].get('title', 'Unknown')}")
        print(f"JSON file size: {os.path.getsize(output_filename) if os.path.exists(output_filename) else 'Unknown'} bytes")
        print("\nFirst 500 characters of extracted text:")
        print(result['extracted_text']['full_text'][:500] + "...")
    else:
        print(f"Extraction failed: {result['error']}")
        # Still save the error result for analysis
        print(f"Error result saved to: {output_filename}")