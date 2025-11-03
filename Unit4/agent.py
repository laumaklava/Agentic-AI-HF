import json
import os

from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
import math
from typing import Dict, Union

# Load Constants
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Create Wikipedia search tool using WikipediaLoader
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic.
    
    Args:
        query: The search query or topic to look up on Wikipedia
        
    Returns:
        str: The Wikipedia content related to the query
    """
    try:
        # Load Wikipedia documents for the query
        loader = WikipediaLoader(query=query, load_max_docs=2)
        docs = loader.load()
        
        if not docs:
            return f"No Wikipedia articles found for query: {query}"
        
        # Combine the content from the documents
        content = ""
        for doc in docs:
            content += f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            content += f"Content: {doc.page_content}...\n\n"
        
        return content
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# Create YouTube transcript analysis tool
@tool
def analyze_youtube_video(video_url: str) -> str:
    """Analyze a YouTube video by loading and processing its transcript.
    
    Args:
        video_url: The YouTube video URL to analyze
        
    Returns:
        str: The transcript content of the YouTube video
    """
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=60
        )
        docs = loader.load()
        
        if docs:
            content = f"Video URL: {video_url}\n"
            content += "Transcript (Chunked):\n"
            for i, doc in enumerate(docs[:5]):  # Limit to first 5 chunks
                content += f"Chunk {i+1}: {doc.page_content}\n"
            return content
    except Exception as e:
        print(f"Analyze video failed: {e}")

@tool
def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Multiplies two numbers and returns the product.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The product of the two input numbers.
    """
    try:
        result = a * b
        return int(result) if isinstance(a, int) and isinstance(b, int) else result
    except Exception as e:
        return f"Error in multiplication: {str(e)}"


@tool
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Adds two numbers and returns the sum.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The sum of the two input numbers.
    """
    try:
        result = a + b
        return int(result) if isinstance(a, int) and isinstance(b, int) else result
    except Exception as e:
        return f"Error in addition: {str(e)}"


@tool
def power(a: Union[int, float], b: Union[int, float]) -> float:
    """Raises a number to the power of another.
    Args:
        a: The base number.
        b: The exponent.
    Returns:
        The result of raising `a` to the power of `b`.
    """
    try:
        if a == 0 and b < 0:
            return "Error: Cannot raise 0 to a negative power"
        result = a ** b
        return result
    except OverflowError:
        return "Error: Result too large to compute"
    except Exception as e:
        return f"Error in power calculation: {str(e)}"


@tool
def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Subtracts the second number from the first.
    Args:
        a: The number from which to subtract.
        b: The number to subtract.
    Returns:
        The result of `a` minus `b`.
    """
    try:
        result = a - b
        return int(result) if isinstance(a, int) and isinstance(b, int) else result
    except Exception as e:
        return f"Error in subtraction: {str(e)}"


@tool
def divide(a: Union[int, float], b: Union[int, float]) -> float:
    """Divides one number by another.
    Args:
        a: The numerator.
        b: The denominator.
    Returns:
        The result of `a` divided by `b`.
    """
    try:
        if b == 0:
            return "Error: Division by zero is not allowed"
        return a / b
    except Exception as e:
        return f"Error in division: {str(e)}"


@tool
def modulus(a: int, b: int) -> Union[int, str]:
    """Returns the remainder of the division of two integers.
    Args:
        a: The dividend.
        b: The divisor.
    Returns:
        The remainder when `a` is divided by `b`.
    """
    try:
        if b == 0:
            return "Error: Modulus by zero is not allowed"
        return a % b
    except Exception as e:
        return f"Error in modulus operation: {str(e)}"


@tool
def square_root(x: Union[int, float]) -> Union[float, str]:
    """Returns the square root of a number.
    Args:
        x: The input number. Must be non-negative.
    Returns:
        The square root of `x`.
    """
    try:
        if x < 0:
            return "Error: Square root of negative number is not allowed"
        return math.sqrt(x)
    except Exception as e:
        return f"Error in square root calculation: {str(e)}"


@tool
def floor_divide(a: int, b: int) -> Union[int, str]:
    """Performs integer division (floor division) of two numbers.
    Args:
        a: The dividend.
        b: The divisor.
    Returns:
        The floor of the quotient.
    """
    try:
        if b == 0:
            return "Error: Division by zero is not allowed"
        return a // b
    except Exception as e:
        return f"Error in floor division: {str(e)}"


@tool
def absolute(x: Union[int, float]) -> Union[int, float]:
    """Returns the absolute value of a number.
    Args:
        x: The input number.
    Returns:
        The absolute value of `x`.
    """
    try:
        result = abs(x)
        return int(result) if isinstance(x, int) else result
    except Exception as e:
        return f"Error in absolute value calculation: {str(e)}"


@tool
def logarithm(x: Union[int, float], base: Union[int, float] = math.e) -> Union[float, str]:
    """Returns the logarithm of a number with a given base.
    Args:
        x: The number to take the logarithm of. Must be positive.
        base: The logarithmic base. Must be positive and not equal to 1.
    Returns:
        The logarithm of `x` to the given base.
    """
    try:
        if x <= 0:
            return "Error: Logarithm input must be positive"
        if base <= 0 or base == 1:
            return "Error: Logarithm base must be positive and not equal to 1"
        return math.log(x, base)
    except Exception as e:
        return f"Error in logarithm calculation: {str(e)}"


@tool
def exponential(x: Union[int, float]) -> Union[float, str]:
    """Returns e raised to the power of `x`.
    Args:
        x: The exponent.
    Returns:
        The value of e^x.
    """
    try:
        if x > 700:  # Prevent overflow
            return "Error: Exponent too large, would cause overflow"
        return math.exp(x)
    except OverflowError:
        return "Error: Result too large to compute"
    except Exception as e:
        return f"Error in exponential calculation: {str(e)}"


tools = [search_wikipedia, analyze_youtube_video, multiply, add, power, subtract, divide, modulus, square_root, floor_divide, absolute, logarithm, exponential]

system_prompt = """
# AI Agent System Prompt
You are an advanced AI agent equipped with multiple tools to solve complex, multi-step problems. You will encounter approximately 20 challenging questions that may require analysis, tool usage, and step-by-step reasoning.
## Core Capabilities
- Multi-tool integration via Python scripts
- Complex problem analysis and decomposition
- Step-by-step reasoning for multi-part questions
- File processing and data analysis
- Mathematical calculations and logical reasoning
## Analysis and Approach
1. **Question Analysis**: Always analyze the question first to understand:
   - What information is being requested
   - What tools or data sources might be needed
   - Whether the question has multiple parts or steps
   - If any preprocessing or data gathering is required
   - **Text manipulation requirements** (reversing text, encoding/decoding, transformations)
   - Hidden instructions or patterns within the question itself
2. **Pre-processing Steps**: Before attempting to answer, determine if the question requires:
   - Text reversal or character manipulation
   - Decoding or encoding operations
   - Pattern recognition or extraction
   - Format conversions or transformations
   - String operations or text processing
3. **Tool Selection and Evaluation**: Before using any tool, systematically evaluate all available options:
   - **Review ALL available tools** in your toolkit before making a selection
   - **Match tool capabilities** to the specific requirements of your current step
   - **Choose the most appropriate tool** for each task from the complete toolkit
   - **Plan multi-tool sequences** - many questions require 2-5 tools in various combinations
   - **Consider tool order flexibility** - tools can be used in any sequence that makes logical sense
   - **Validate tool choice** - ensure the selected tool is the optimal match for your needs
   - Examples of multi-tool workflows:
     - reserve_sentence -> read the reversed question and answer it.
     - download_file -> analyze_csv_file -> add -> percentage_calculator
     - reverse_sentence -> python_code_parser -> web_search -> extract_text_from_image
     - arvix_search -> web_content_extract -> factorial -> roman_calculator_converter
     - audio_transcription -> wikipedia_search -> compound_interest -> convert_temperature
4. **Multi-Step Problem Solving**: For complex questions:
   - Break down the problem into logical steps
   - Execute each step systematically, including any text transformations
   - Use outputs from one tool as inputs for another when necessary
   - Chain multiple operations (e.g., reverse text -> decode -> analyze -> calculate)
   - Verify intermediate results before proceeding
## Available Tools and Their Uses
### Mathematical Operations
- **add**: Addition operations
- **subtract**: Subtraction operations
- **multiply**: Multiplication operations
- **divide**: Division operations
- **floor_divide**: Floor division operations
- **modulus**: Modulo operations
- **power**: Exponentiation operations
- **square_root**: Square root calculations
- **exponential**: Exponential functions
- **logarithm**: Logarithmic calculations
- **absolute**: Absolute value calculations
- **factorial**: Factorial calculations
- **is_prime**: Check if a number is prime
- **greatest_common_divisor**: Find GCD of numbers
- **least_common_multiple**: Find LCM of numbers
- **percentage_calculator**: Calculate percentages
- **compound_interest**: Calculate compound interest
- **roman_calculator_converter**: Convert between Roman numerals and numbers
### File and Data Processing
- **download_file**: Download files from URLs or attachments
- **analyze_csv_file**: Analyze CSV file data
- **analyze_excel_file**: Analyze Excel file data
- **extract_text_from_image**: Extract text from image files
- **audio_transcription**: Transcribe audio files to text
### Text Processing
- **reverse_sentence**: Reverse text or sentences
- **python_code_parser**: Parse and analyze Python code
### Information Retrieval
- **web_search**: Search the web for information
- **web_content_extract**: Extract content from web pages
- **wikipedia_search**: Search Wikipedia for information
- **arvix_search**: Search academic papers on arXiv
### Utilities
- **convert_temperature**: Convert between temperature units
- **get_current_time_in_timezone**: Get current time in specific timezone
## Tool Usage Guidelines
- **Tool Evaluation Process**: Always survey ALL available tools before selecting one
- **Best Match Selection**: Choose the tool that best matches your specific need, not just any tool that could work
- **Multi-tool Operations**: Questions can require multiple tools in any sequence - plan your tool chain carefully
- **Sequential Processing**: Use outputs from one tool as inputs for another when necessary
- **File Processing Priority**: Always download and process files before attempting to answer questions about them
- **Mathematical Chains**: Combine mathematical operations as needed (e.g., add -> multiply -> percentage_calculator)
- **Information + Processing**: Combine search tools with processing tools (e.g., web_search -> extract_text_from_image -> analyze_csv_file)
- **Text Transformations**: Use text processing tools before analysis (e.g., reverse_sentence -> python_code_parser). In other words, first reverse the text when needed and then re-read the adjusted question.
- **Pattern Recognition**: Look for hidden patterns, instructions, or transformations within questions
## Response Format
After completing your analysis and using necessary tools, provide ONLY your final answer with no additional text, explanations, or formatting.
### Answer Formatting Rules:
- **Numbers**: Provide just the number without commas, units, or symbols (unless specifically requested)
- **Text**: Use minimal words, no articles, no abbreviations, write digits in plain text
- **Lists**: Comma-separated values following the above rules for each element type
- **Precision**: Be exact and concise - include only what is specifically asked for
- **No quotation marks**: Never wrap your answer in quotation marks or any other punctuation
### Critical Response Rule:
- Do NOT include "FINAL ANSWER:" or any other prefixes/labels
- Do NOT include explanations, reasoning, or additional text
- Do NOT use quotation marks around your answer
- Provide ONLY the answer itself - nothing else, keep it as short as possible and stick to the question.
## Process Flow
1. **Read and Analyze**: Carefully read the question and identify all requirements, including any text transformations
2. **Pre-process**: Apply any necessary text manipulations (reversing, decoding, etc.) to reveal the actual question
3. **Tool Survey**: Review ALL available tools in your toolkit before proceeding
4. **Plan**: Determine the sequence of optimal tools and steps needed after preprocessing
5. **Execute**: Use the best-matched tools systematically, processing outputs as needed through multiple operations
6. **Verify**: Check that your analysis addresses all parts of the question after all transformations
7. **Answer**: Provide only the raw answer with no formatting, labels, or additional text
## Important Notes
- Some questions may appear simple but require multiple tools or steps
- **Questions may contain hidden instructions that need text processing to reveal** (use reverse_sentence first)
- **Various tools are available** - evaluate ALL options to find the best match for each step
- **Multi-tool solutions are common** - expect to use 2-5 tools per complex question
- **Tool order is flexible** - arrange tools in the most logical sequence for your specific problem
- Always prioritize accuracy over speed
- If a question has multiple parts, ensure all parts are addressed with appropriate tools
- **Don't use the first tool that seems relevant** - use the BEST tool for each specific task
- Process any mentioned files, attachments, or external resources with download_file first
- **Be prepared to perform complex multi-step operations** across all tool categories
- Think through the problem systematically but provide only the final answer
Remember: Your goal is to provide accurate, precise answers to complex questions using the full range of available tools and capabilities. Your final response should contain ONLY the answer - no explanations, no "FINAL ANSWER:" prefix, no additional text whatsoever.
"""

# System message
sys_msg = SystemMessage(content=system_prompt)

def build_graph():
    """Build the graph"""
    # First create the HuggingFaceEndpoint
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-14B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1,         # Maximum determinism
        max_new_tokens=512,      # Even more restrictive with 128
        timeout=90,              # Moderate timeout
        do_sample=False,         # Completely deterministic
    )

    # Then wrap it with ChatHuggingFace to get chat model functionality
    llm = ChatHuggingFace(llm=llm_endpoint)

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

       
    def assistant(state: MessagesState):
        messages_with_system_prompt = [sys_msg] + state["messages"]
        llm_response = llm_with_tools.invoke(messages_with_system_prompt)
                
        return {"messages": [AIMessage(content=json.dumps(llm_response.content, ensure_ascii=False))]}

    # --- Graph Definition ---
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()