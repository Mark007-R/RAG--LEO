import os
import pickle
import json
import shutil
import logging
from pathlib import Path
from typing import Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_dirs(dirs: Optional[List[str]] = None) -> None:
    """
    Ensure required directories exist.
    
    Args:
        dirs: List of directory paths to create. If None, creates default directories.
    """
    default_dirs = ['uploads', 'indexes', 'metadata', 'logs', 'temp']
    directories = dirs or default_dirs
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


def save_pickle(obj: Any, path: str) -> None:
    """
    Save object to pickle file with error handling.
    
    Args:
        obj: Python object to serialize
        path: File path to save to
        
    Raises:
        IOError: If save fails
    """
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(path)
        logger.debug(f"Saved pickle to {path} ({file_size} bytes)")
        
    except Exception as e:
        logger.error(f"Failed to save pickle to {path}: {e}")
        raise IOError(f"Pickle save failed: {str(e)}")


def load_pickle(path: str) -> Any:
    """
    Load object from pickle file with error handling.
    
    Args:
        path: File path to load from
        
    Returns:
        Deserialized Python object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If load fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        
        logger.debug(f"Loaded pickle from {path}")
        return obj
        
    except Exception as e:
        logger.error(f"Failed to load pickle from {path}: {e}")
        raise IOError(f"Pickle load failed: {str(e)}")


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to serialize (must be JSON-serializable)
        path: File path to save to
        indent: Indentation for pretty printing
        
    Raises:
        IOError: If save fails
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.debug(f"Saved JSON to {path}")
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise IOError(f"JSON save failed: {str(e)}")


def load_json(path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: File path to load from
        
    Returns:
        Deserialized data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If load fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON from {path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise IOError(f"JSON load failed: {str(e)}")


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def cleanup_old_files(directory: str, days: int = 7) -> int:
    """
    Remove files older than specified days.
    
    Args:
        directory: Directory to clean
        days: Files older than this many days will be removed
        
    Returns:
        Number of files removed
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return 0
    
    try:
        removed_count = 0
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath):
                file_time = os.path.getmtime(filepath)
                
                if file_time < cutoff_time:
                    os.remove(filepath)
                    removed_count += 1
                    logger.info(f"Removed old file: {filepath}")
        
        logger.info(f"Cleanup completed: removed {removed_count} files from {directory}")
        return removed_count
        
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")
        return 0


def delete_document_files(doc_id: str) -> dict:
    """
    Delete all files associated with a document ID.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Dictionary with deletion statistics
    """
    deleted = {
        'uploads': 0,
        'indexes': 0,
        'metadata': 0
    }
    
    try:
        # Remove uploads
        upload_path = Path('uploads')
        for file_path in upload_path.glob(f"{doc_id}__*"):
            os.remove(file_path)
            deleted['uploads'] += 1
            logger.info(f"Deleted upload: {file_path}")
        
        # Remove index
        index_path = Path('indexes') / f"{doc_id}.index"
        if index_path.exists():
            os.remove(index_path)
            deleted['indexes'] += 1
            logger.info(f"Deleted index: {index_path}")
        
        # Remove metadata
        metadata_path = Path('metadata') / f"{doc_id}_chunks.pkl"
        if metadata_path.exists():
            os.remove(metadata_path)
            deleted['metadata'] += 1
            logger.info(f"Deleted metadata: {metadata_path}")
        
        logger.info(f"Deleted all files for doc_id: {doc_id}")
        
    except Exception as e:
        logger.error(f"Error deleting files for doc_id {doc_id}: {e}")
    
    return deleted


def get_directory_stats(directory: str) -> dict:
    """
    Get statistics about a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Dictionary with file count and total size
    """
    if not os.path.exists(directory):
        return {'file_count': 0, 'total_size': 0, 'total_size_formatted': '0 B'}
    
    try:
        file_count = 0
        total_size = 0
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_count += 1
                total_size += os.path.getsize(filepath)
        
        return {
            'file_count': file_count,
            'total_size': total_size,
            'total_size_formatted': format_file_size(total_size)
        }
        
    except Exception as e:
        logger.error(f"Error getting stats for {directory}: {e}")
        return {'file_count': 0, 'total_size': 0, 'total_size_formatted': '0 B'}


def clear_temp_files() -> int:
    """
    Clear all temporary files.
    
    Returns:
        Number of files removed
    """
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        return 0
    
    try:
        removed = 0
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                removed += 1
        
        logger.info(f"Cleared {removed} temporary files")
        return removed
        
    except Exception as e:
        logger.error(f"Error clearing temp files: {e}")
        return 0


def backup_file(source_path: str, backup_dir: str = 'backups') -> Optional[str]:
    """
    Create a backup copy of a file with timestamp.
    
    Args:
        source_path: Path to file to backup
        backup_dir: Directory to store backups
        
    Returns:
        Path to backup file, or None if backup failed
    """
    if not os.path.exists(source_path):
        logger.warning(f"Source file not found: {source_path}")
        return None
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.basename(source_path)
        backup_path = os.path.join(backup_dir, f"{timestamp}_{filename}")
        
        shutil.copy2(source_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        return backup_path
        
    except Exception as e:
        logger.error(f"Failed to create backup of {source_path}: {e}")
        return None


def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Name of file to validate
        allowed_extensions: Set of allowed extensions (e.g., {'pdf', 'txt'})
        
    Returns:
        True if file type is allowed
    """
    if '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in allowed_extensions


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace problematic characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Trim to max length while preserving extension
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename


def get_all_documents() -> dict:
    """
    Get metadata for all indexed documents.
    
    Returns:
        Dictionary mapping doc_id to metadata
    """
    documents = {}
    
    try:
        if not os.path.exists('indexes'):
            return documents
        
        for filename in os.listdir('indexes'):
            if filename.endswith('.index'):
                doc_id = filename[:-6]  # Remove '.index'
                
                index_path = os.path.join('indexes', filename)
                chunks_path = os.path.join('metadata', f'{doc_id}_chunks.pkl')
                
                documents[doc_id] = {
                    'index_exists': os.path.exists(index_path),
                    'chunks_exist': os.path.exists(chunks_path),
                    'index_size': get_file_size(index_path),
                    'created': datetime.fromtimestamp(os.path.getctime(index_path)).isoformat()
                }
        
        logger.debug(f"Found {len(documents)} indexed documents")
        
    except Exception as e:
        logger.error(f"Error getting document list: {e}")
    
    return documents