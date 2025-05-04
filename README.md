# Weather AI Assistant

A sassy, fun weather assistant built with Streamlit and the Groq LLM API.

## Features

- Get weather information for any location worldwide
- Enjoy sassy, humorous responses with city-specific facts
- Integration with Groq's LLama3-70B model
- Support for custom API keys with unlimited usage
- Free tier with limited queries for non-API key users

## How to Use

1. Enter your question about weather for any location in the text box
2. Click "Send" to get a response
3. Optionally enter your own Groq API key for unlimited queries

## Setup

### Local Development

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```
   streamlit run frontend/app.py
   ```

### Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app, selecting your forked repository
4. Set the main file path to `frontend/app.py`
5. Add your Groq API key as a secret:
   - In the app settings, find the "Secrets" section
   - Add the following:
     ```
     GROQ_API_KEY=your_api_key_here
     ```
6. Deploy the app!

## API Key

You need a Groq API key to use this app with unlimited queries. You can:

1. Get a key from [Groq's website](https://console.groq.com/keys)
2. Enter it in the app interface
3. Or set it as a secret in Streamlit Cloud for the default key

## Technologies Used

- Streamlit for the UI
- Groq LLM API (with LLama3-70B model)
- wttr.in for weather data 