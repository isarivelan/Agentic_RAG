## Tool Calling with Azure OpenAI and LlamaIndex
`This script demonstrates how to use Azure OpenAI with LlamaIndex for tool calling and Retrieval-Augmented Generation (RAG).`

Prerequisites
Python 3.7 or higher
Required Python packages:
llama-index-core
llama-index-llms-azure-openai
llama-index-embeddings-azure-openai
dotenv
nest_asyncio
You can install the required packages using the following command:


Setup
Environment Variables:

Create a .env file in the root directory of your project.

Add your Azure OpenAI credentials to the .env file:


Load Environment Variables:

The script uses dotenv to load environment variables.
Apply nest_asyncio:

nest_asyncio is applied for compatibility with asynchronous operations.
Script Overview
Imports
The script imports necessary libraries and modules, including FunctionTool from llama_index.core.tools, AzureOpenAI, and AzureOpenAIEmbedding.

Functions
setup_azure_openai():

Sets up Azure OpenAI LLM and embedding models.
Initializes AzureOpenAI and AzureOpenAIEmbedding with the provided credentials.
create_simple_tools():

Creates simple mathematical function tools (add and mystery).
test_simple_tools(llm, tools):

Tests the simple function tools using the Azure OpenAI LLM.
load_and_process_documents(file_path: Optional[str] = None):

Loads and processes documents for RAG.
Uses SimpleDirectoryReader to load data from a specified PDF file.
create_vector_index(nodes):

Creates a vector index from nodes using VectorStoreIndex.
create_vector_query_tool(vector_index):

Creates a vector query tool with page filtering.
create_summary_tool(nodes):

Creates a summary tool using SummaryIndex.
test_vector_query_tool(llm, vector_query_tool):

Tests the vector query tool with a sample query.
test_multiple_tools(llm, tools, queries):

Tests multiple tools with various queries.
Main Function
The main() function orchestrates the entire process:

Sets up Azure OpenAI.
Tests simple tools.
Loads and processes documents.
Creates vector index and tools.
Tests vector query tool.
Tests multiple tools with various queries.
Running the Script
To run the script, execute the following command:


Ensure that your .env file is correctly configured with your Azure OpenAI credentials.
