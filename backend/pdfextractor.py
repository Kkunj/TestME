# pdf_extractor.py
"""
PDF Text Extraction System for Educational Content Processing
Designed for extracting text from lecture slides and textbooks
Enhanced with comprehensive logging for deployment debugging
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pypdf
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import sys
import traceback
import platform


class PDFExtractor:
    """
    A comprehensive PDF text extraction class optimized for educational content.
    Handles various PDF layouts and provides multiple extraction modes.
    Enhanced with detailed logging for deployment debugging.
    """
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize the PDF extractor with logging configuration.
        
        Args:
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file (str): Optional log file path for persistent logging
        """
        self.setup_logging(log_level, log_file)
        self.logger = logging.getLogger(__name__)
        self.log_system_info()
        
    def setup_logging(self, level: str, log_file: Optional[str] = None) -> None:
        """Configure comprehensive logging for the extractor."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            try:
                # Ensure log directory exists
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                print(f"✅ Log file configured: {log_file}")
            except Exception as e:
                print(f"⚠️ Failed to setup file logging: {str(e)}")
    
    def log_system_info(self) -> None:
        """Log system and environment information for debugging."""
        self.logger.info("="*50)
        self.logger.info("PDF EXTRACTOR INITIALIZATION")
        self.logger.info("="*50)
        
        try:
            # System information
            self.logger.info(f"Platform: {platform.platform()}")
            self.logger.info(f"Python version: {sys.version}")
            self.logger.info(f"Python executable: {sys.executable}")
            self.logger.info(f"Current working directory: {os.getcwd()}")
            self.logger.info(f"User: {os.getenv('USER', 'Unknown')}")
            self.logger.info(f"Home directory: {os.getenv('HOME', 'Unknown')}")
            
            # PyPDF version
            self.logger.info(f"PyPDF version: {pypdf.__version__}")
            
            # Environment variables (relevant ones)
            env_vars = ['PATH', 'PYTHONPATH', 'HOME', 'USER', 'PWD']
            for var in env_vars:
                value = os.getenv(var, 'Not set')
                # Truncate long paths for readability
                if len(value) > 200:
                    value = value[:100] + "..." + value[-97:]
                self.logger.info(f"ENV {var}: {value}")
                
        except Exception as e:
            self.logger.error(f"Failed to log system info: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def extract_text_from_pdf(
        self, 
        file_path: str, 
        extraction_mode: str = "simple",
        page_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, any]:
        """
        Extract text from PDF file with various extraction modes.
        Enhanced with comprehensive logging.
        
        Args:
            file_path (str): Path to the PDF file
            extraction_mode (str): Extraction mode ('simple', 'layout_aware', 'structured')
            page_range (tuple): Optional tuple (start_page, end_page) for partial extraction
            
        Returns:
            Dict containing extracted text, metadata, and extraction info
        """
        self.logger.info(f"Starting PDF extraction for: {file_path}")
        self.logger.info(f"Extraction mode: {extraction_mode}")
        self.logger.info(f"Page range: {page_range}")
        
        try:
            # Log file path details
            abs_path = os.path.abspath(file_path)
            self.logger.info(f"Absolute path: {abs_path}")
            
            # Validate file path
            if not self._validate_pdf_file(file_path):
                error_msg = f"Invalid PDF file path or file doesn't exist: {file_path}"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg)
            
            self.logger.info("✅ File validation passed")
            
            # Read PDF
            self.logger.info("Attempting to read PDF...")
            reader = PdfReader(file_path)
            self.logger.info(f"✅ PDF reader created successfully")
            
            # Log PDF basic info
            total_pages = len(reader.pages)
            self.logger.info(f"Total pages in PDF: {total_pages}")
            
            # Get PDF metadata
            self.logger.info("Extracting PDF metadata...")
            metadata = self._extract_metadata(reader)
            self.logger.info(f"✅ Metadata extracted: {len(metadata)} fields")
            
            # Determine page range
            start_page, end_page = self._determine_page_range(page_range, total_pages)
            self.logger.info(f"Processing pages {start_page + 1} to {end_page} (0-indexed: {start_page}-{end_page-1})")
            
            # Extract text based on mode
            self.logger.info(f"Starting {extraction_mode} extraction...")
            
            if extraction_mode == "simple":
                extracted_text = self._simple_extraction(reader, start_page, end_page)
            elif extraction_mode == "layout_aware":
                extracted_text = self._layout_aware_extraction(reader, start_page, end_page)
            elif extraction_mode == "structured":
                extracted_text = self._structured_extraction(reader, start_page, end_page)
            else:
                error_msg = f"Unknown extraction mode: {extraction_mode}"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg)
            
            self.logger.info(f"✅ Text extraction completed")
            self.logger.info(f"Total words extracted: {extracted_text.get('total_words', 0)}")
            self.logger.info(f"Full text length: {len(extracted_text.get('full_text', ''))}")
            
            # Check if extraction returned empty text
            if not extracted_text.get('full_text', '').strip():
                self.logger.warning("⚠️ WARNING: Extracted text is empty!")
                self.logger.warning("This could indicate:")
                self.logger.warning("1. PDF contains only images/scanned content")
                self.logger.warning("2. PDF is password protected")
                self.logger.warning("3. PDF has unusual encoding")
                self.logger.warning("4. PyPDF compatibility issues")
            
            response = self._create_success_response(
                extracted_text, metadata, total_pages, extraction_mode, start_page, end_page
            )
            
            self.logger.info("✅ PDF extraction completed successfully")
            return response
            
        except PdfReadError as e:
            error_msg = f"PDF reading error: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"PDF file: {file_path}")
            self.logger.error(traceback.format_exc())
            return self._create_error_response(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during extraction: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"PDF file: {file_path}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(traceback.format_exc())
            return self._create_error_response(error_msg)
    
    def _validate_pdf_file(self, file_path: str) -> bool:
        """Validate if the file exists and is a PDF with detailed logging."""
        self.logger.info(f"Validating PDF file: {file_path}")
        
        try:
            path = Path(file_path)
            
            # Check if path exists
            if not path.exists():
                self.logger.error(f"❌ File does not exist: {file_path}")
                self.logger.error(f"Current working directory: {os.getcwd()}")
                # List files in current directory for debugging
                try:
                    files = os.listdir(os.getcwd())
                    self.logger.info(f"Files in current directory: {files[:10]}...")  # First 10 files
                except Exception as list_error:
                    self.logger.error(f"Could not list current directory: {str(list_error)}")
                return False
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                self.logger.error(f"❌ File is not readable: {file_path}")
                return False
            
            # Check file extension
            if path.suffix.lower() != '.pdf':
                self.logger.error(f"❌ File is not a PDF (extension: {path.suffix}): {file_path}")
                return False
            
            # Check file size
            file_size = path.stat().st_size
            self.logger.info(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            if file_size == 0:
                self.logger.error("❌ PDF file is empty (0 bytes)")
                return False
            
            # Try to peek at file header
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(5)
                    if not header.startswith(b'%PDF-'):
                        self.logger.error(f"❌ File doesn't have PDF header: {header}")
                        return False
                    else:
                        self.logger.info(f"✅ Valid PDF header found: {header}")
            except Exception as header_error:
                self.logger.error(f"Failed to read file header: {str(header_error)}")
                return False
            
            self.logger.info("✅ PDF file validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during file validation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _extract_metadata(self, reader: PdfReader) -> Dict[str, any]:
        """Extract metadata from PDF with detailed logging."""
        self.logger.info("Extracting PDF metadata...")
        metadata = {}
        
        try:
            if reader.metadata is None:
                self.logger.warning("PDF metadata is None")
                return {'error': 'No metadata available'}
            
            raw_metadata = reader.metadata
            self.logger.info(f"Raw metadata keys: {list(raw_metadata.keys())}")
            
            metadata = {
                'title': raw_metadata.get('/Title', 'Unknown'),
                'author': raw_metadata.get('/Author', 'Unknown'),
                'subject': raw_metadata.get('/Subject', 'Unknown'),
                'creator': raw_metadata.get('/Creator', 'Unknown'),
                'producer': raw_metadata.get('/Producer', 'Unknown'),
                'creation_date': str(raw_metadata.get('/CreationDate', 'Unknown')),
                'modification_date': str(raw_metadata.get('/ModDate', 'Unknown'))
            }
            
            self.logger.info("✅ Metadata extraction successful")
            for key, value in metadata.items():
                self.logger.debug(f"Metadata {key}: {value}")
                
        except Exception as e:
            error_msg = f"Could not extract metadata: {str(e)}"
            self.logger.warning(error_msg)
            self.logger.warning(traceback.format_exc())
            metadata = {'error': 'Metadata extraction failed'}
        
        return metadata
    
    def _determine_page_range(self, page_range: Optional[Tuple[int, int]], total_pages: int) -> Tuple[int, int]:
        """Determine the actual page range to process with logging."""
        if page_range is None:
            self.logger.info(f"No page range specified, processing all {total_pages} pages")
            return 0, total_pages
        
        start_page = max(0, page_range[0])
        end_page = min(total_pages, page_range[1])
        
        self.logger.info(f"Page range adjusted: requested({page_range[0]}, {page_range[1]}) -> actual({start_page}, {end_page})")
        
        if start_page >= end_page:
            self.logger.warning(f"Invalid page range: start_page({start_page}) >= end_page({end_page})")
        
        return start_page, end_page
    
    def _simple_extraction(self, reader: PdfReader, start_page: int, end_page: int) -> Dict[str, any]:
        """Simple text extraction without layout preservation with detailed logging."""
        self.logger.info(f"Starting simple extraction for pages {start_page} to {end_page-1}")
        
        pages_text = []
        full_text = ""
        successful_pages = 0
        failed_pages = 0
        
        for page_num in range(start_page, end_page):
            self.logger.debug(f"Processing page {page_num + 1}")
            
            try:
                page = reader.pages[page_num]
                self.logger.debug(f"Page object created for page {page_num + 1}")
                
                page_text = page.extract_text()
                text_length = len(page_text)
                word_count = len(page_text.split())
                
                self.logger.debug(f"Page {page_num + 1}: extracted {text_length} characters, {word_count} words")
                
                if not page_text.strip():
                    self.logger.warning(f"Page {page_num + 1} returned empty text")
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'word_count': word_count,
                    'char_count': text_length
                })
                full_text += page_text + "\n\n"
                successful_pages += 1
                
            except Exception as e:
                error_msg = f"Error extracting text from page {page_num + 1}: {str(e)}"
                self.logger.warning(error_msg)
                self.logger.warning(traceback.format_exc())
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': '',
                    'error': str(e),
                    'word_count': 0,
                    'char_count': 0
                })
                failed_pages += 1
        
        total_words = len(full_text.split())
        self.logger.info(f"Simple extraction completed: {successful_pages} successful, {failed_pages} failed")
        self.logger.info(f"Total text length: {len(full_text)} characters, {total_words} words")
        
        return {
            'full_text': full_text.strip(),
            'pages': pages_text,
            'total_words': total_words,
            'successful_pages': successful_pages,
            'failed_pages': failed_pages
        }
    
    def _layout_aware_extraction(self, reader: PdfReader, start_page: int, end_page: int) -> Dict[str, any]:
        """Layout-aware text extraction with detailed logging."""
        self.logger.info(f"Starting layout-aware extraction for pages {start_page} to {end_page-1}")
        
        pages_text = []
        full_text = ""
        successful_pages = 0
        failed_pages = 0
        fallback_pages = 0
        
        for page_num in range(start_page, end_page):
            self.logger.debug(f"Processing page {page_num + 1} with layout-aware extraction")
            
            try:
                page = reader.pages[page_num]
                
                # Try layout mode first
                try:
                    page_text = page.extract_text(extraction_mode="layout")
                    extraction_method = 'layout_aware'
                    self.logger.debug(f"Page {page_num + 1}: layout extraction successful")
                except Exception as layout_error:
                    self.logger.warning(f"Layout extraction failed for page {page_num + 1}: {str(layout_error)}")
                    # Fallback to simple extraction
                    page_text = page.extract_text()
                    extraction_method = 'fallback_simple'
                    fallback_pages += 1
                    self.logger.debug(f"Page {page_num + 1}: using fallback simple extraction")
                
                # Clean up the extracted text
                cleaned_text = self._clean_extracted_text(page_text)
                text_length = len(cleaned_text)
                word_count = len(cleaned_text.split())
                
                self.logger.debug(f"Page {page_num + 1}: {text_length} characters, {word_count} words after cleaning")
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'word_count': word_count,
                    'char_count': text_length,
                    'extraction_mode': extraction_method
                })
                full_text += cleaned_text + "\n\n"
                successful_pages += 1
                
            except Exception as e:
                error_msg = f"Error in layout-aware extraction from page {page_num + 1}: {str(e)}"
                self.logger.warning(error_msg)
                self.logger.warning(traceback.format_exc())
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': '',
                    'error': str(e),
                    'word_count': 0,
                    'char_count': 0
                })
                failed_pages += 1
        
        total_words = len(full_text.split())
        self.logger.info(f"Layout-aware extraction completed: {successful_pages} successful, {failed_pages} failed, {fallback_pages} fallback")
        self.logger.info(f"Total text length: {len(full_text)} characters, {total_words} words")
        
        return {
            'full_text': full_text.strip(),
            'pages': pages_text,
            'total_words': total_words,
            'successful_pages': successful_pages,
            'failed_pages': failed_pages,
            'fallback_pages': fallback_pages
        }
    
    def _structured_extraction(self, reader: PdfReader, start_page: int, end_page: int) -> Dict[str, any]:
        """Structured extraction with additional text analysis and detailed logging."""
        self.logger.info(f"Starting structured extraction for pages {start_page} to {end_page-1}")
        
        pages_text = []
        full_text = ""
        successful_pages = 0
        failed_pages = 0
        document_structure = {
            'headings': [],
            'paragraphs': [],
            'bullet_points': [],
            'numbered_lists': []
        }
        
        for page_num in range(start_page, end_page):
            self.logger.debug(f"Processing page {page_num + 1} with structured extraction")
            
            try:
                page = reader.pages[page_num]
                page_text = page.extract_text(extraction_mode="layout")
                cleaned_text = self._clean_extracted_text(page_text)
                
                self.logger.debug(f"Page {page_num + 1}: {len(cleaned_text)} characters after cleaning")
                
                # Analyze text structure
                page_structure = self._analyze_text_structure(cleaned_text, page_num + 1)
                
                # Log structure analysis results
                structure_counts = {k: len(v) for k, v in page_structure.items()}
                self.logger.debug(f"Page {page_num + 1} structure: {structure_counts}")
                
                # Merge with document structure
                for key in document_structure:
                    document_structure[key].extend(page_structure[key])
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text),
                    'structure': page_structure
                })
                full_text += cleaned_text + "\n\n"
                successful_pages += 1
                
            except Exception as e:
                error_msg = f"Error in structured extraction from page {page_num + 1}: {str(e)}"
                self.logger.warning(error_msg)
                self.logger.warning(traceback.format_exc())
                
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': '',
                    'error': str(e),
                    'word_count': 0,
                    'char_count': 0
                })
                failed_pages += 1
        
        total_words = len(full_text.split())
        
        # Log overall document structure
        total_structure = {k: len(v) for k, v in document_structure.items()}
        self.logger.info(f"Structured extraction completed: {successful_pages} successful, {failed_pages} failed")
        self.logger.info(f"Total text length: {len(full_text)} characters, {total_words} words")
        self.logger.info(f"Document structure: {total_structure}")
        
        return {
            'full_text': full_text.strip(),
            'pages': pages_text,
            'total_words': total_words,
            'successful_pages': successful_pages,
            'failed_pages': failed_pages,
            'document_structure': document_structure
        }
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text with logging."""
        if not text:
            self.logger.debug("Text cleaning: input text is empty")
            return ""
        
        original_length = len(text)
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = ' '.join(line.split())
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        final_length = len(cleaned_text)
        
        self.logger.debug(f"Text cleaning: {original_length} -> {final_length} characters")
        
        return cleaned_text
    
    def _analyze_text_structure(self, text: str, page_num: int) -> Dict[str, List[Dict]]:
        """Analyze text structure with detailed logging."""
        if not text:
            self.logger.debug(f"Page {page_num}: No text to analyze")
            return {
                'headings': [],
                'paragraphs': [],
                'bullet_points': [],
                'numbered_lists': []
            }
        
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
            
            # Classify line type
            if self._is_heading(line):
                structure['headings'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1
                })
            elif self._is_bullet_point(line):
                structure['bullet_points'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1
                })
            elif self._is_numbered_list(line):
                structure['numbered_lists'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1
                })
            else:
                structure['paragraphs'].append({
                    'text': line,
                    'page': page_num,
                    'line_number': i + 1,
                    'word_count': len(line.split())
                })
        
        # Log structure analysis results
        counts = {k: len(v) for k, v in structure.items()}
        self.logger.debug(f"Page {page_num} structure analysis: {counts}")
        
        return structure
    
    def _is_heading(self, line: str) -> bool:
        """Determine if a line is likely a heading."""
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
        """Create a successful extraction response with logging."""
        response = {
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
        
        self.logger.info("✅ Success response created")
        return response
    
    def _create_error_response(self, error_message: str) -> Dict[str, any]:
        """Create an error response with logging."""
        response = {
            'success': False,
            'extracted_text': None,
            'metadata': None,
            'extraction_info': None,
            'error': error_message
        }
        
        self.logger.error(f"❌ Error response created: {error_message}")
        return response
    
    def extract_text_from_multiple_pdfs(
        self, 
        file_paths: List[str], 
        extraction_mode: str = "simple"
    ) -> Dict[str, Dict[str, any]]:
        """
        Extract text from multiple PDF files with detailed logging.
        
        Args:
            file_paths (List[str]): List of PDF file paths
            extraction_mode (str): Extraction mode to use for all files
            
        Returns:
            Dict with file paths as keys and extraction results as values
        """
        self.logger.info(f"Starting batch extraction of {len(file_paths)} PDFs")
        self.logger.info(f"Files to process: {file_paths}")
        
        results = {}
        successful_files = 0
        failed_files = 0
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"Processing file {i}/{len(file_paths)}: {file_path}")
            
            result = self.extract_text_from_pdf(file_path, extraction_mode)
            results[file_path] = result
            
            if result['success']:
                successful_files += 1
                self.logger.info(f"✅ File {i} completed successfully")
            else:
                failed_files += 1
                self.logger.error(f"❌ File {i} failed: {result['error']}")
        
        self.logger.info(f"Batch extraction completed: {successful_files} successful, {failed_files} failed")
        return results
    
    def get_pdf_info(self, file_path: str) -> Dict[str, any]:
        """
        Get basic information about a PDF file with detailed logging.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict containing PDF information
        """
        self.logger.info(f"Getting PDF info for: {file_path}")
        
        try:
            if not self._validate_pdf_file(file_path):
                error_msg = 'Invalid PDF file path or file doesn\'t exist'
                self.logger.error(error_msg)
                return {'error': error_msg}
            
            reader = PdfReader(file_path)
            metadata = self._extract_metadata(reader)
            file_size = os.path.getsize(file_path)
            
            info = {
                'success': True,
                'file_path': file_path,
                'total_pages': len(reader.pages),
                'metadata': metadata,
                'file_size': file_size,
                'file_size_mb': round(file_size / 1024 / 1024, 2),
                'error': None
            }
            
            self.logger.info(f"✅ PDF info extracted successfully")
            self.logger.info(f"Pages: {info['total_pages']}, Size: {info['file_size_mb']} MB")
            
            return info
            
        except Exception as e:
            error_msg = f"Failed to get PDF info: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': error_msg
            }


# Enhanced main functions with logging support
def extract_pdf_text(
    file_path: str, 
    extraction_mode: str = "simple",
    page_range: Optional[Tuple[int, int]] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> Dict[str, any]:
    """
    Main function to extract text from PDF - designed for external access.
    Enhanced with comprehensive logging for deployment debugging.
    
    Args:
        file_path (str): Path to the PDF file
        extraction_mode (str): 'simple', 'layout_aware', or 'structured'
        page_range (tuple): Optional (start_page, end_page) for partial extraction
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str): Optional log file path for persistent logging
        
    Returns:
        Dict containing extraction results
    """
    # Create extractor with logging
    extractor = PDFExtractor(log_level=log_level, log_file=log_file)
    
    # Log function call details
    extractor.logger.info("="*60)
    extractor.logger.info("EXTRACT_PDF_TEXT FUNCTION CALLED")
    extractor.logger.info("="*60)
    extractor.logger.info(f"Function arguments:")
    extractor.logger.info(f"  file_path: {file_path}")
    extractor.logger.info(f"  extraction_mode: {extraction_mode}")
    extractor.logger.info(f"  page_range: {page_range}")
    extractor.logger.info(f"  log_level: {log_level}")
    extractor.logger.info(f"  log_file: {log_file}")
    
    try:
        result = extractor.extract_text_from_pdf(file_path, extraction_mode, page_range)
        
        # Log final result summary
        if result['success']:
            extractor.logger.info("="*60)
            extractor.logger.info("✅ EXTRACTION COMPLETED SUCCESSFULLY")
            extractor.logger.info("="*60)
            extractor.logger.info(f"Pages processed: {result['extraction_info']['pages_processed']}")
            extractor.logger.info(f"Total words: {result['extracted_text']['total_words']}")
            extractor.logger.info(f"Text length: {len(result['extracted_text']['full_text'])} characters")
            
            # Warning if text is suspiciously short
            if result['extracted_text']['total_words'] < 10:
                extractor.logger.warning("⚠️ WARNING: Very few words extracted - this might indicate an issue!")
                
        else:
            extractor.logger.error("="*60)
            extractor.logger.error("❌ EXTRACTION FAILED")
            extractor.logger.error("="*60)
            extractor.logger.error(f"Error: {result['error']}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in extract_pdf_text: {str(e)}"
        extractor.logger.error(error_msg)
        extractor.logger.error(traceback.format_exc())
        
        return {
            'success': False,
            'extracted_text': None,
            'metadata': None,
            'extraction_info': None,
            'error': error_msg
        }


def extract_multiple_pdfs(
    file_paths: List[str], 
    extraction_mode: str = "simple",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> Dict[str, Dict[str, any]]:
    """
    Extract text from multiple PDF files with enhanced logging.
    
    Args:
        file_paths (List[str]): List of PDF file paths
        extraction_mode (str): Extraction mode to use
        log_level (str): Logging level
        log_file (str): Optional log file path
        
    Returns:
        Dict with results for each file
    """
    extractor = PDFExtractor(log_level=log_level, log_file=log_file)
    
    extractor.logger.info("="*60)
    extractor.logger.info("EXTRACT_MULTIPLE_PDFS FUNCTION CALLED")
    extractor.logger.info("="*60)
    extractor.logger.info(f"Processing {len(file_paths)} files with mode: {extraction_mode}")
    
    return extractor.extract_text_from_multiple_pdfs(file_paths, extraction_mode)


def get_pdf_info(
    file_path: str, 
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> Dict[str, any]:
    """
    Get basic PDF information with enhanced logging.
    
    Args:
        file_path (str): Path to the PDF file
        log_level (str): Logging level
        log_file (str): Optional log file path
        
    Returns:
        Dict containing PDF information
    """
    extractor = PDFExtractor(log_level=log_level, log_file=log_file)
    
    extractor.logger.info("="*60)
    extractor.logger.info("GET_PDF_INFO FUNCTION CALLED")
    extractor.logger.info("="*60)
    
    return extractor.get_pdf_info(file_path)


def diagnose_pdf_issues(
    file_path: str,
    log_file: Optional[str] = None
) -> Dict[str, any]:
    """
    Comprehensive diagnostic function to identify PDF extraction issues.
    This function performs multiple checks to help identify deployment problems.
    
    Args:
        file_path (str): Path to the PDF file
        log_file (str): Optional log file path
        
    Returns:
        Dict containing diagnostic information
    """
    # Always use DEBUG level for diagnostics
    extractor = PDFExtractor(log_level="DEBUG", log_file=log_file)
    
    extractor.logger.info("="*60)
    extractor.logger.info("PDF DIAGNOSTIC MODE STARTED")
    extractor.logger.info("="*60)
    
    diagnostic_results = {
        'file_path': file_path,
        'timestamp': None,
        'system_info': {},
        'file_checks': {},
        'extraction_tests': {},
        'recommendations': []
    }
    
    try:
        import datetime
        diagnostic_results['timestamp'] = datetime.datetime.now().isoformat()
        
        # System information
        diagnostic_results['system_info'] = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'pypdf_version': pypdf.__version__,
            'working_directory': os.getcwd(),
            'user': os.getenv('USER', 'Unknown')
        }
        
        # File system checks
        extractor.logger.info("🔍 PERFORMING FILE SYSTEM CHECKS")
        
        file_checks = {}
        try:
            abs_path = os.path.abspath(file_path)
            file_checks['absolute_path'] = abs_path
            file_checks['exists'] = os.path.exists(file_path)
            file_checks['is_file'] = os.path.isfile(file_path) if os.path.exists(file_path) else False
            file_checks['readable'] = os.access(file_path, os.R_OK) if os.path.exists(file_path) else False
            file_checks['size'] = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Check file permissions in detail
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                file_checks['permissions'] = oct(stat_info.st_mode)[-3:]
                file_checks['owner_readable'] = bool(stat_info.st_mode & 0o400)
                file_checks['group_readable'] = bool(stat_info.st_mode & 0o040)
                file_checks['other_readable'] = bool(stat_info.st_mode & 0o004)
            
            # Try to read first few bytes
            if file_checks['readable']:
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(10)
                        file_checks['header'] = header.hex()
                        file_checks['is_pdf_header'] = header.startswith(b'%PDF-')
                except Exception as header_error:
                    file_checks['header_error'] = str(header_error)
            
        except Exception as file_error:
            file_checks['error'] = str(file_error)
            extractor.logger.error(f"File system check error: {str(file_error)}")
        
        diagnostic_results['file_checks'] = file_checks
        
        # PDF extraction tests
        extractor.logger.info("🔍 PERFORMING EXTRACTION TESTS")
        
        extraction_tests = {}
        
        if file_checks.get('exists', False) and file_checks.get('readable', False):
            # Test 1: Basic PDF reading
            try:
                reader = PdfReader(file_path)
                extraction_tests['pdf_readable'] = True
                extraction_tests['total_pages'] = len(reader.pages)
                
                if len(reader.pages) > 0:
                    # Test first page extraction with different methods
                    first_page = reader.pages[0]
                    
                    # Simple extraction
                    try:
                        simple_text = first_page.extract_text()
                        extraction_tests['simple_extraction'] = {
                            'success': True,
                            'text_length': len(simple_text),
                            'word_count': len(simple_text.split()),
                            'has_content': bool(simple_text.strip())
                        }
                    except Exception as simple_error:
                        extraction_tests['simple_extraction'] = {
                            'success': False,
                            'error': str(simple_error)
                        }
                    
                    # Layout-aware extraction
                    try:
                        layout_text = first_page.extract_text(extraction_mode="layout")
                        extraction_tests['layout_extraction'] = {
                            'success': True,
                            'text_length': len(layout_text),
                            'word_count': len(layout_text.split()),
                            'has_content': bool(layout_text.strip())
                        }
                    except Exception as layout_error:
                        extraction_tests['layout_extraction'] = {
                            'success': False,
                            'error': str(layout_error)
                        }
                
            except Exception as pdf_error:
                extraction_tests['pdf_readable'] = False
                extraction_tests['pdf_error'] = str(pdf_error)
                extractor.logger.error(f"PDF reading error: {str(pdf_error)}")
        
        else:
            extraction_tests['skipped'] = "File not accessible for testing"
        
        diagnostic_results['extraction_tests'] = extraction_tests
        
        # Generate recommendations
        recommendations = []
        
        if not file_checks.get('exists', False):
            recommendations.append("❌ File does not exist - check file path and ensure file is uploaded/accessible")
        
        if not file_checks.get('readable', False):
            recommendations.append("❌ File is not readable - check file permissions")
        
        if file_checks.get('size', 0) == 0:
            recommendations.append("❌ File is empty (0 bytes) - ensure file was transferred correctly")
        
        if not file_checks.get('is_pdf_header', False):
            recommendations.append("❌ File doesn't have valid PDF header - file may be corrupted or not a PDF")
        
        if extraction_tests.get('pdf_readable', False):
            simple_test = extraction_tests.get('simple_extraction', {})
            layout_test = extraction_tests.get('layout_extraction', {})
            
            if not simple_test.get('success', False) and not layout_test.get('success', False):
                recommendations.append("❌ Both extraction methods failed - PDF may be password protected or severely corrupted")
            
            elif simple_test.get('success', False) and not simple_test.get('has_content', False):
                recommendations.append("⚠️ PDF extraction successful but no text found - PDF may contain only images or scanned content")
        
        if not recommendations:
            recommendations.append("✅ No obvious issues detected - PDF should extract normally")
        
        diagnostic_results['recommendations'] = recommendations
        
        # Log summary
        extractor.logger.info("="*60)
        extractor.logger.info("🏁 DIAGNOSTIC SUMMARY")
        extractor.logger.info("="*60)
        
        for rec in recommendations:
            if rec.startswith("❌"):
                extractor.logger.error(rec)
            elif rec.startswith("⚠️"):
                extractor.logger.warning(rec)
            else:
                extractor.logger.info(rec)
        
        return diagnostic_results
        
    except Exception as diag_error:
        error_msg = f"Diagnostic function error: {str(diag_error)}"
        extractor.logger.error(error_msg)
        extractor.logger.error(traceback.format_exc())
        
        diagnostic_results['error'] = error_msg
        return diagnostic_results


# Example usage and testing with enhanced logging
# if __name__ == "__main__":
#     import json
#     from datetime import datetime
    
#     # Configuration
#     sample_pdf_path = "STD9.pdf"
#     log_file_path = f"pdf_extraction_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
#     print("🚀 Starting PDF Extraction with Enhanced Logging")
#     print(f"📝 Debug log will be saved to: {log_file_path}")
#     print("="*60)
    
#     # First, run diagnostics
#     print("🔍 Running PDF diagnostics...")
#     diagnostic_result = diagnose_pdf_issues(sample_pdf_path, log_file=log_file_path)
    
#     # Save diagnostic results
#     diag_filename = f"pdf_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     try:
#         with open(diag_filename, 'w', encoding='utf-8') as f:
#             json.dump(diagnostic_result, f, indent=2, ensure_ascii=False, default=str)
#         print(f"📊 Diagnostic results saved to: {diag_filename}")
#     except Exception as save_error:
#         print(f"⚠️ Could not save diagnostic results: {str(save_error)}")
    
#     # Print recommendations
#     print("\n💡 RECOMMENDATIONS:")
#     for rec in diagnostic_result.get('recommendations', []):
#         print(f"   {rec}")
    
#     print("\n" + "="*60)
    
#     # Proceed with extraction if basic checks pass
#     if diagnostic_result.get('file_checks', {}).get('exists', False):
#         print("📖 Proceeding with text extraction...")
        
#         # Run extraction with detailed logging
#         result = extract_pdf_text(
#             sample_pdf_path, 
#             extraction_mode="structured",
#             log_level="DEBUG",  # Use DEBUG for maximum detail
#             log_file=log_file_path
#         )
        
#         # Generate output filename with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         pdf_filename = os.path.splitext(os.path.basename(sample_pdf_path))[0]
#         output_filename = f"extraction_result_{pdf_filename}_{timestamp}.json"
        
#         # Save result to JSON file
#         try:
#             with open(output_filename, 'w', encoding='utf-8') as json_file:
#                 json.dump(result, json_file, indent=2, ensure_ascii=False, default=str)
#             print(f"💾 Extraction result saved to: {output_filename}")
#         except Exception as e:
#             print(f"❌ Failed to save JSON file: {str(e)}")
        
#         # Display summary
#         if result['success']:
#             print(f"✅ Successfully extracted text from {result['extraction_info']['pages_processed']} pages")
#             print(f"📊 Total words: {result['extracted_text']['total_words']}")
#             print(f"📄 Document title: {result['metadata'].get('title', 'Unknown')}")
#             if os.path.exists(output_filename):
#                 print(f"📦 JSON file size: {os.path.getsize(output_filename)} bytes")
            
#             # Show preview if text was extracted
#             full_text = result['extracted_text']['full_text']
#             if full_text and len(full_text.strip()) > 0:
#                 print(f"\n📝 First 300 characters of extracted text:")
#                 print("-" * 40)
#                 print(full_text[:300] + "..." if len(full_text) > 300 else full_text)
#                 print("-" * 40)
#             else:
#                 print("⚠️ WARNING: No text content was extracted!")
                
#         else:
#             print(f"❌ Extraction failed: {result['error']}")
    
#     else:
#         print("❌ Skipping extraction due to file access issues")
    
#     print(f"\n📋 Complete debug log available at: {log_file_path}")
#     print("🏁 Done!")
