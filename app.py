import streamlit as st
import requests
import uuid
import json
import os
import re
from datetime import datetime, timedelta
from openai import OpenAI, APIError, AuthenticationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'show_key_removed_message' not in st.session_state:
    st.session_state.show_key_removed_message = False
if 'using_api_key' not in st.session_state:
    st.session_state.using_api_key = False

# Default API key from environment
DEFAULT_API_KEY = os.getenv("GROQ_API_KEY")

# System prompt for the LLM
system_prompt = """
You are a sassy weather assistant who ONLY provides weather information. If users ask about anything else, you should roast them playfully and remind them to stick to weather queries.

Your tasks:
1. Analyze the user's query to determine if it's weather-related
2. If weather-related, extract the city name and provide weather information
3. If not weather-related, roast the user creatively
4. For weather queries, follow this format:
   - Share an interesting fact about the city
   - Provide the weather information
   - Add a quirky suggestion based on the weather condition

Example responses:
- For weather queries:
  "Ah, Tokyo! Did you know it's the world's largest metropolitan area? The weather is currently 25°C and sunny. Perfect weather for a romantic date! Oh wait, you're probably going to enjoy it alone with your cat. At least your cat won't ghost you!"

- For non-weather queries:
  "Oh honey, I'm a weather assistant, not your personal Google! Stick to asking about the weather, and maybe I'll tell you if you should bring an umbrella or not. 😏"

Remember:
- Be sassy and humorous
- Include city-specific facts
- Add weather-based suggestions with a touch of humor
- Roast users who ask non-weather questions
- Keep responses engaging and fun

Available Tools:
- get_weather: Takes a city name as input and returns current weather

Rules:
    - Follow the Output JSON Format.
    - Carefully analyse the user query

Output JSON Format:
{{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function"
}}
"""

# Helper functions from backend
def get_weather(city: str):
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        return "Weather data unavailable"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

def create_openai_client(api_key=None):
    """Create an OpenAI client with the given API key or default"""
    return OpenAI(
        api_key=api_key or DEFAULT_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

def format_llm_response(llm_response, query):
    """Handle malformed JSON from LLM and return properly formatted response"""
    try:
        # Try to parse the response as JSON
        parsed_response = json.loads(llm_response)
        return parsed_response
    except json.JSONDecodeError:
        # If JSON parsing fails, extract content and reformat
        try:
            # Look for patterns in malformed responses
            if '"step":' in llm_response and '"function":' in llm_response:
                # Extract content between step and function
                content_start = llm_response.find('"step":')
                content_end = llm_response.find('"function":')
                
                if content_start != -1 and content_end != -1:
                    # Find the actual content after "step": "..."
                    step_value_start = llm_response.find('"', content_start + 7) + 1
                    step_value_end = llm_response.find('"', step_value_start)
                    step_value = llm_response[step_value_start:step_value_end]
                    
                    # Extract the content in between
                    raw_content = llm_response[step_value_end+1:content_end].strip()
                    # Clean up the content
                    if raw_content.startswith(','):
                        raw_content = raw_content[1:].strip()
                    
                    # Find the function and input values if present
                    function_match = re.search(r'"function":\s*"([^"]+)"', llm_response)
                    input_match = re.search(r'"input":\s*"([^"]+)"', llm_response)
                    
                    function_value = function_match.group(1) if function_match else "get_weather"
                    input_value = input_match.group(1) if input_match else ""
                    
                    # Create a properly formatted response
                    return {
                        "step": step_value,
                        "content": raw_content.strip('" \n,'),
                        "function": function_value,
                        "input": input_value
                    }
            
            # Default fallback if pattern matching fails
            return {
                "content": f"I had trouble with that weather query. Please try asking about the weather in a specific city."
            }
        except Exception as e:
            # Last resort fallback
            return {
                "content": f"I had trouble processing your query about '{query}'. Please try asking about the weather in a different way."
            }

def get_weather_info(query, session_id, api_key=None, usage_count=0):
    # Check if API key is provided and if it's not empty
    has_api_key = api_key and api_key.strip() != ""
    
    # Check usage limit for non-API key users
    if not has_api_key and usage_count >= 5:
        return {"error": "Usage limit exceeded. Please provide your API key for unlimited access."}, 429
    
    # Determine which API key to use
    api_key_to_use = None
    if has_api_key:
        api_key_to_use = api_key
    else:
        api_key_to_use = DEFAULT_API_KEY
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        # Create client with appropriate API key
        client = create_openai_client(api_key_to_use)
        
        # Try to make the API call
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            response_format={"type": "json_object"},
            messages=messages
        )
        
        # Safely parse the response and handle malformed JSON
        try:
            parsed_response = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Handle malformed JSON
            parsed_response = format_llm_response(response.choices[0].message.content, query)
        
        # Calculate remaining usage
        remaining_usage = 999999 if has_api_key else max(0, 5 - usage_count)
        
        return {
            "response": parsed_response,
            "remaining_usage": remaining_usage,
            "using_custom_key": has_api_key
        }, 200
    except AuthenticationError:
        # Handle invalid API key
        if has_api_key:
            return {"error": "Invalid API key. Please check your API key and try again."}, 401
        else:
            # If the default key fails
            return {"error": "Server API key configuration error. Please try again later or use your own API key."}, 500
    except APIError as e:
        # Check for JSON generation error specifically
        error_message = str(e)
        if "json_validate_failed" in error_message or "Failed to generate JSON" in error_message:
            # Extract the failed generation if available to create a better response
            failed_json = None
            try:
                if hasattr(e, 'response') and e.response:
                    error_data = e.response.json().get('error', {})
                    failed_generation = error_data.get('failed_generation')
                    if failed_generation:
                        # Try to extract useful content from the failed generation
                        content_match = re.search(r'"content":\s*"([^"]+)"', failed_generation)
                        if content_match:
                            error_content = content_match.group(1)
                            # If there's a pattern like get_weather("City"), extract city and get weather
                            city_match = re.search(r'get_weather\("([^"]+)"\)', error_content)
                            if city_match:
                                city = city_match.group(1)
                                weather = get_weather(city)
                                # Construct a useful response with the actual weather
                                modified_content = error_content.replace('str(get_weather("' + city + '"))', weather)
                                return {
                                    "response": {"content": modified_content},
                                    "remaining_usage": 999999 if has_api_key else max(0, 5 - usage_count),
                                    "using_custom_key": has_api_key
                                }, 200
            except:
                pass
                
            # Fallback friendly message if we couldn't extract useful information
            return {
                "response": {"content": "I understood your weather query, but had trouble processing it. Please try asking in a simpler way or try a different location."},
                "remaining_usage": 999999 if has_api_key else max(0, 5 - usage_count),
                "using_custom_key": has_api_key,
                "no_usage_deduction": True  # Flag to indicate we shouldn't deduct usage
            }, 200
        
        # Handle other API errors
        return {"error": f"API Error: {str(e)}"}, 500
    except Exception as e:
        # Handle other errors
        return {"error": f"An error occurred: {str(e)}"}, 500

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .weather-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .response-message {
        background-color: white;
        color: #333;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Mobile optimization */
    @media (max-width: 768px) {
        .main, .st-emotion-cache-19lc5he, .st-emotion-cache-uf99v8 {
            background-color: #f5f5f5 !important;
        }
        body {
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .response-message {
            background-color: white !important;
            color: #333 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# JavaScript for localStorage with proper Streamlit communication
st.components.v1.html(
    """
    <script>
    // Get session ID from localStorage or create new one
    function getSessionId() {
        let sessionId = localStorage.getItem('weather_app_session_id');
        if (!sessionId) {
            sessionId = crypto.randomUUID();
            localStorage.setItem('weather_app_session_id', sessionId);
        }
        return sessionId;
    }
    
    // Get usage count from localStorage
    function getUsageCount() {
        let count = localStorage.getItem('weather_app_usage_count');
        return count ? parseInt(count) : 0;
    }
    
    // Set usage count in localStorage
    function setUsageCount(count) {
        localStorage.setItem('weather_app_usage_count', count);
    }
    
    // Wait for Streamlit to be fully loaded
    window.addEventListener('load', function() {
        // Use URL parameters to communicate with Streamlit
        const sessionId = getSessionId();
        const usageCount = getUsageCount();
        
        // Set these values directly in a hidden element that Streamlit can read
        document.getElementById('storage-data').textContent = JSON.stringify({
            sessionId: sessionId,
            usageCount: usageCount
        });
        
        // Force refresh on initial load to ensure parameters are passed
        const baseUrl = window.location.href.split('?')[0];
        if (!window.location.href.includes('storage_initialized=true')) {
            const newUrl = baseUrl + '?storage_initialized=true';
            window.history.replaceState({}, '', newUrl);
            setTimeout(function() {
                window.location.reload();
            }, 100);
        }
    });
    </script>
    <div id="storage-data" style="display:none;"></div>
    """,
    height=0
)

# Try to read localStorage data from the hidden element
storage_reader = st.components.v1.html(
    """
    <script>
    // Function to read data from localStorage
    function readStorage() {
        const sessionId = localStorage.getItem('weather_app_session_id') || '';
        const usageCount = localStorage.getItem('weather_app_usage_count') || '0';
        
        // Output the data in a format that can be captured
        document.write(JSON.stringify({
            sessionId: sessionId,
            usageCount: parseInt(usageCount)
        }));
    }
    
    // Execute immediately
    readStorage();
    </script>
    """,
    height=0
)

# Try to extract localStorage data
try:
    # If successful rerun, storage_reader should contain our data
    if storage_reader and not storage_reader.strip().startswith('<'):
        storage_data = json.loads(storage_reader)
        
        # Update session state from localStorage
        if 'sessionId' in storage_data and storage_data['sessionId']:
            st.session_state.session_id = storage_data['sessionId']
        
        if 'usageCount' in storage_data:
            st.session_state.usage_count = int(storage_data['usageCount'])
        
        st.session_state.initialized = True
except:
    # Fall back to URL parameters if needed
    if 'sessionId' in st.query_params and 'usageCount' in st.query_params:
        st.session_state.session_id = st.query_params['sessionId']
        st.session_state.usage_count = int(st.query_params['usageCount'])
        st.session_state.initialized = True

# Title and description
st.title("🌤️ Weather AI Assistant")
st.markdown("### Your sassy weather companion!")

# Check if we need to display the API key removed message
if 'show_key_removed_message' in st.session_state and st.session_state.show_key_removed_message:
    st.warning("API key has been removed. You're now using limited access.")
    st.session_state.show_key_removed_message = False

# Show usage information
if st.session_state.api_key:
    st.success("✨ Unlimited access with your API key!")
else:
    st.warning(f"⚠️ You have {max(0, 5 - st.session_state.usage_count)} free queries remaining!")

# API Key input
api_key = st.text_input("Enter your Groq API key for unlimited access (optional)", type="password")
if api_key and api_key != st.session_state.api_key:
    # New API key entered
    st.session_state.api_key = api_key
    st.session_state.using_api_key = True
    st.rerun()
elif api_key == "" and st.session_state.api_key:
    # User has cleared the API key
    st.session_state.api_key = None
    st.session_state.show_key_removed_message = True
    st.session_state.using_api_key = False
    st.rerun()

# User input
user_input = st.text_input("Ask me anything about weather in any language!", placeholder="e.g., How's the weather in Tokyo? or What's the temperature in New York?")

def update_local_storage(count):
    """Helper function to update localStorage with current usage count"""
    return st.components.v1.html(
        f"""
        <script>
        try {{
            // Update localStorage with the latest count
            localStorage.setItem('weather_app_usage_count', {count});
            console.log('Updated usage count in localStorage to: {count}');
        }} catch (e) {{
            console.error('Error updating localStorage:', e);
        }}
        </script>
        """,
        height=0
    )

if st.button("Send"):
    if not user_input:
        st.error("Please enter your question!")
    else:
        try:
            # Increment usage count
            st.session_state.usage_count += 1
            
            # Update usage count in localStorage
            update_local_storage(st.session_state.usage_count)
            
            # Ensure api_key is a string (not None)
            api_key_to_send = st.session_state.api_key or ""
            
            # Call weather info function directly instead of making an API request
            data, status_code = get_weather_info(
                query=user_input,
                session_id=st.session_state.session_id,
                api_key=api_key_to_send,
                usage_count=st.session_state.usage_count
            )
            
            if status_code == 429:
                st.error("You've exceeded your free queries! Please enter an API key for unlimited access.")
            elif status_code == 401:
                error_detail = data.get("error", "Invalid API key")
                st.error(f"🚫 {error_detail}")
                
                # Reset API key in session state
                st.session_state.api_key = None
                st.session_state.using_api_key = False
                
                # Revert usage count increment
                st.session_state.usage_count -= 1
                update_local_storage(st.session_state.usage_count)
            elif status_code != 200:
                error_detail = data.get("error", "Unknown error")
                st.error(f"Error: {error_detail}")
                
                # Revert usage count increment
                st.session_state.usage_count -= 1
                update_local_storage(st.session_state.usage_count)
            else:
                try:
                    remaining = data.get("remaining_usage", 0)
                    using_custom_key = data.get("using_custom_key", False)
                    
                    # Check if we shouldn't deduct usage (for gracefully handled errors)
                    if data.get("no_usage_deduction", False) and not using_custom_key:
                        # Revert usage count increment for gracefully handled errors
                        st.session_state.usage_count -= 1
                        update_local_storage(st.session_state.usage_count)
                        # Recalculate remaining
                        remaining = max(0, 5 - st.session_state.usage_count)
                    
                    # Update the API key status in session state
                    st.session_state.using_api_key = using_custom_key
                    
                    # Extract the content from the response
                    response_data = data["response"]
                    if "content" in response_data:
                        # Standard content format
                        response_content = response_data["content"]
                    elif "step" in response_data and "content" not in response_data:
                        # Malformed response with missing content field
                        response_content = "I couldn't process that weather query properly. Please try again."
                    else:
                        # Fallback to using the entire response as content
                        response_content = str(response_data)
                    
                    # Display the response
                    st.markdown(f'<div class="response-message">{response_content}</div>', unsafe_allow_html=True)
                    
                    # Show remaining usage
                    if not using_custom_key:
                        st.info(f"Remaining free queries: {remaining}")
                    else:
                        st.success("Using your API key for unlimited access")
                except Exception as e:
                    st.error(f"Unable to process response: {str(e)}")
                    # Revert usage count increment
                    st.session_state.usage_count -= 1
                    update_local_storage(st.session_state.usage_count)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
            # Revert usage count increment on error
            st.session_state.usage_count -= 1
            update_local_storage(st.session_state.usage_count)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Hritik Jain") 