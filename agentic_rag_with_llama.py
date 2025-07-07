"""
Tool Calling with Azure OpenAI and LlamaIndex
This script demonstrates how to use Azure OpenAI with LlamaIndex for tool calling and RAG.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
import nest_asyncio

# Load environment variables
load_dotenv()

# Apply nest_asyncio for compatibility
nest_asyncio.apply()

# Import required libraries
try:
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.azure_openai import AzureOpenAI
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.vector_stores import MetadataFilters, FilterCondition
    from llama_index.core.tools import QueryEngineTool
    print("✓ All LlamaIndex imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install required packages using:")
    print("pip install llama-index llama-index-llms-azure-openai llama-index-embeddings-azure-openai")
    exit(1)


def setup_azure_openai():
    """Setup Azure OpenAI LLM and embedding models"""
    # Check environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key or not endpoint:
        raise ValueError("Missing Azure OpenAI credentials. Please check your .env file.")
    
    # Initialize Azure OpenAI LLM
    llm = AzureOpenAI(
        model="gpt-4o",  # Replace with your model name
        api_key=api_key,
        engine="gpt-4o",
        azure_endpoint=endpoint,
        api_version="2024-02-15-preview",
        temperature=0
    )
    
    # Initialize Azure OpenAI embedding
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
          # Replace with your embedding model
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-02-15-preview",
    )
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model


def create_simple_tools():
    """Create simple mathematical function tools"""
    def add(x: int, y: int) -> int:
        """Adds two integers together."""
        return x + y

    def mystery(x: int, y: int) -> int: 
        """Mystery function that operates on top of two numbers."""
        return (x + y) * (x + y)

    add_tool = FunctionTool.from_defaults(fn=add)
    mystery_tool = FunctionTool.from_defaults(fn=mystery)
    
    return add_tool, mystery_tool


def test_simple_tools(llm, tools):
    """Test simple function tools"""
    print("=== Testing Simple Tools ===")
    try:
        response = llm.predict_and_call(
            tools, 
            "Tell me the output of the mystery function on 2 and 9", 
            verbose=True
        )
        print(f"Response: {response}")
        return response
    except Exception as e:
        print(f"Error in simple tools test: {e}")
        return None


def load_and_process_documents(file_path: Optional[str] = None):
    """Load and process documents for RAG"""
    # Default file path - update this to your PDF location
    if file_path is None:
        file_path = r"C:\Users\isarivelan.mani\Downloads\5488_MetaGPT_Meta_Programming_.pdf"
    
    print(f"=== Loading documents from {file_path} ===")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        
        # Try to find any PDF files in the current directory
        current_dir = os.getcwd()
        pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]
        
        if pdf_files:
            print(f"Found PDF files in current directory: {pdf_files}")
            file_path = os.path.join(current_dir, pdf_files[0])
            print(f"Using: {file_path}")
        else:
            print("No PDF files found. Please ensure a PDF file exists.")
            return None, None
    
    try:
        # Load documents
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        print(f"Loaded {len(documents)} documents")
        
        # Split into nodes
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} nodes")
        
        # Display sample node
        if nodes:
            print(f"Sample node content: {nodes[0].get_content(metadata_mode='all')[:200]}...")
        
        return documents, nodes
    
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None, None


def create_vector_index(nodes):
    """Create vector index from nodes"""
    print("=== Creating Vector Index ===")
    try:
        vector_index = VectorStoreIndex(nodes)
        return vector_index
    except Exception as e:
        print(f"Error creating vector index: {e}")
        return None


def create_vector_query_tool(vector_index):
    """Create a vector query tool with page filtering"""
    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """Perform a vector search over an index.
        
        Args:
            query (str): the string query to be embedded.
            page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        Returns:
            str: The response from the query engine
        """
        try:
            # Handle case when page_numbers is None or empty
            if page_numbers:
                metadata_dicts = [
                    {"key": "page_label", "value": p} for p in page_numbers
                ]
                
                query_engine = vector_index.as_query_engine(
                    similarity_top_k=2,
                    filters=MetadataFilters.from_dicts(
                        metadata_dicts,
                        condition=FilterCondition.OR
                    )
                )
            else:
                # No filtering - search all pages
                query_engine = vector_index.as_query_engine(
                    similarity_top_k=2
                )
            
            response = query_engine.query(query)
            return str(response)
        
        except Exception as e:
            return f"Error in vector query: {str(e)}"

    vector_query_tool = FunctionTool.from_defaults(
        name="vector_tool",
        fn=vector_query
    )
    
    return vector_query_tool


def create_summary_tool(nodes):
    """Create a summary tool"""
    print("=== Creating Summary Tool ===")
    try:
        summary_index = SummaryIndex(nodes)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        summary_tool = QueryEngineTool.from_defaults(
            name="summary_tool",
            query_engine=summary_query_engine,
            description="Useful if you want to get a summary of the document"
        )
        return summary_tool
    except Exception as e:
        print(f"Error creating summary tool: {e}")
        return None


def test_vector_query_tool(llm, vector_query_tool):
    """Test vector query tool"""
    print("=== Testing Vector Query Tool ===")
    
    try:
        # Test with page filtering
        response = llm.predict_and_call(
            [vector_query_tool], 
            "What are the high-level results described in the document?", 
            verbose=True
        )
        print(f"Response: {response}")
        
        # Print source metadata if available
        if hasattr(response, 'source_nodes'):
            print("\nSource nodes metadata:")
            for n in response.source_nodes:
                print(n.metadata)
        
        return response
    
    except Exception as e:
        print(f"Error in vector query test: {e}")
        return None


def test_multiple_tools(llm, tools, queries):
    """Test multiple tools with various queries"""
    print("=== Testing Multiple Tools ===")
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        try:
            response = llm.predict_and_call(
                tools, 
                query, 
                verbose=True
            )
            print(f"Response: {response}")
            
            # Print source metadata if available
            if hasattr(response, 'source_nodes'):
                print("Source nodes metadata:")
                for n in response.source_nodes:
                    print(n.metadata)
        except Exception as e:
            print(f"Error processing query: {e}")


def main():
    """Main function to run the tool calling examples"""
    print("Starting Tool Calling with Azure OpenAI Demo")
    print("=" * 50)
    
    # Setup Azure OpenAI
    try:
        llm, embed_model = setup_azure_openai()
        print("✓ Azure OpenAI setup complete")
    except Exception as e:
        print(f"✗ Error setting up Azure OpenAI: {e}")
        print("Please check your .env file and Azure OpenAI credentials")
        return
    
    # Test simple tools
    print("\n" + "=" * 30)
    add_tool, mystery_tool = create_simple_tools()
    simple_response = test_simple_tools(llm, [add_tool, mystery_tool])
    
    if simple_response is None:
        print("Simple tools test failed. Continuing with document tests...")
    
    # Load and process documents
    print("\n" + "=" * 30)
    documents, nodes = load_and_process_documents()
    
    if nodes is None:
        print("Skipping document-based tests due to missing or unreadable file")
        print("Demo completed with simple tools only!")
        return
    
    # Create vector index and tools
    vector_index = create_vector_index(nodes)
    if vector_index is None:
        print("Failed to create vector index")
        return
    
    vector_query_tool = create_vector_query_tool(vector_index)
    summary_tool = create_summary_tool(nodes)
    
    # Test vector query tool
    print("\n" + "=" * 30)
    test_vector_query_tool(llm, vector_query_tool)
    
    # Test multiple tools with various queries
    print("\n" + "=" * 30)
    test_queries = [
        
        "What are the future directions suggested in the document?",
        "What are the limitations mentioned in the document?",
        "Explain the communication protocols discussed in the document.",
        "What are the key findings of the document?",

    ]
    
    # Only include tools that were successfully created
    available_tools = [tool for tool in [vector_query_tool, summary_tool] if tool is not None]
    
    if available_tools:
        test_multiple_tools(llm, available_tools, test_queries)
    else:
        print("No document tools available for testing")
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    

    main()