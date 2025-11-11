import os
import uuid
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from rag_pipeline import RAGPipeline
from utils import ensure_dirs, save_pickle
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize directories
ensure_dirs()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Global pipeline instance
pipeline = RAGPipeline()

# In-memory document metadata store
DOCS = {}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents."""
    docs_list = [
        {'doc_id': doc_id, 'filename': filename}
        for doc_id, filename in DOCS.items()
    ]
    return jsonify({'documents': docs_list, 'count': len(docs_list)})


@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload and process a PDF document.
    Creates embeddings and FAISS index for retrieval.
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        save_filename = f"{doc_id}__{original_filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
        
        # Save file
        file.save(save_path)
        logger.info(f"Saved file: {save_filename}")

        # Process PDF: extract, chunk, embed, build index
        logger.info(f"Processing document {doc_id}...")
        text = pipeline.extract_text_from_pdf(save_path)
        
        if not text or len(text.strip()) == 0:
            os.remove(save_path)
            return jsonify({'error': 'No text could be extracted from PDF'}), 400
        
        chunks = pipeline.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks from document")
        
        pipeline.build_faiss_index(doc_id)
        logger.info(f"Built FAISS index for document {doc_id}")

        # Store metadata
        DOCS[doc_id] = original_filename

        return jsonify({
            'message': 'Document uploaded and indexed successfully',
            'doc_id': doc_id,
            'filename': original_filename,
            'chunks_count': len(chunks),
            'text_length': len(text)
        }), 201

    except Exception as e:
        logger.error(f"Error during upload: {str(e)}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/ask', methods=['POST'])
def ask():
    """
    Query a document using RAG pipeline.
    Retrieves relevant chunks and generates answer.
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        doc_id = data.get('doc_id', '').strip()
        top_k = data.get('top_k', 4)  # Allow customizable retrieval count

        # Validate inputs
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        if not doc_id:
            return jsonify({'error': 'doc_id parameter is required'}), 400
        if doc_id not in DOCS:
            return jsonify({'error': 'Document not found'}), 404

        logger.info(f"Query received for doc {doc_id}: {query[:100]}...")

        # Load index if needed
        if pipeline.doc_id != doc_id:
            logger.info(f"Loading index for document {doc_id}")
            try:
                pipeline.load_index(doc_id)
            except FileNotFoundError:
                return jsonify({'error': 'Document index not found. Please re-upload the document.'}), 404

        # Retrieve and generate
        retrieved = pipeline.retrieve(query, top_k=top_k)
        answer = pipeline.generate_answer(query, retrieved)

        logger.info(f"Generated answer for query on doc {doc_id}")

        return jsonify({
            'answer': answer,
            'retrieved_chunks': retrieved,
            'doc_id': doc_id,
            'filename': DOCS[doc_id],
            'query': query
        })

    except Exception as e:
        logger.error(f"Error during query: {str(e)}", exc_info=True)
        return jsonify({'error': f'Query failed: {str(e)}'}), 500


@app.route('/document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and its associated index."""
    try:
        if doc_id not in DOCS:
            return jsonify({'error': 'Document not found'}), 404

        filename = DOCS[doc_id]
        
        # Remove uploaded file
        file_pattern = f"{doc_id}__*"
        upload_folder = Path(app.config['UPLOAD_FOLDER'])
        for file_path in upload_folder.glob(file_pattern):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")

        # Remove index files
        index_folder = Path('indexes')
        for index_file in index_folder.glob(f"{doc_id}*"):
            os.remove(index_file)
            logger.info(f"Deleted index: {index_file}")

        # Remove from memory
        del DOCS[doc_id]

        return jsonify({
            'message': 'Document deleted successfully',
            'doc_id': doc_id,
            'filename': filename
        })

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        return jsonify({'error': f'Deletion failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'documents_count': len(DOCS),
        'current_doc_loaded': pipeline.doc_id
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded."""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)