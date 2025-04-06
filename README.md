# AI-Powered News Credibility Verification System

This application is a tool that helps verify the credibility of news articles by cross-referencing them against trusted sources. It uses a combination of AI, web search, and content analysis to provide a credibility score and detailed breakdown of the article's claims.

## Features

*   **Text Input:** Paste news article text directly into the app.
*   **Image Input:** Upload images containing news text, which will be processed using OCR (Optical Character Recognition).
*   **AI-Powered Search:** Uses Gemini to generate relevant search queries based on the article content.
*   **Web Search Integration:** Leverages the Google Custom Search API to find real-world search results.
*   **RAG (Retrieval-Augmented Generation):** Uses Gemini to analyze the article's claims against the retrieved search results.
*   **Credibility Scoring:** Calculates a credibility score based on source reliability, content similarity, and the presence of contradicted or unsupported claims.
*   **Content Analysis:** Provides a detailed analysis of supported, contradicted, and unsupported claims.
*   **Trusted Sources:** Allows customization of trusted news sources.
* **Language processing**: if article is not in english, it will be translated.

## Workflow

1.  **Input:** The user provides a news article either by pasting text or uploading an image.
2.  **OCR (Optional):** If an image is uploaded, EasyOCR extracts the text from the image.
3.  **Language Processing:** Detects the language of the text and translates it to English if needed, using the `translate` library.
4.  **AI-Powered Search Query Generation:** The application uses the Gemini API to generate specific, insightful search queries related to the article's content.
5.  **Web Search:** The Google Custom Search API is used to search the web for the generated queries and gather the top search results.
6.  **RAG (Retrieval-Augmented Generation):** The Gemini API analyzes the original article's claims against the information found in the search results.
7.  **Credibility Scoring:** The application calculates a credibility score based on:
    *   The trustworthiness of the sources found in the search results.
    *   The factual similarity between the article and the search results.
    *   A bonus for supported claims.
    *   Penalties for contradicted or unsupported claims.
8.  **Content Analysis:** Provides a detailed breakdown of supported, contradicted, and unsupported claims.
9.  **Output:** The application displays the credibility score, the content analysis, and the list of reference sources.

## API Setup (for Local Use)

This application requires three API keys:

1.  **Gemini API Key:** For text generation, analysis, and reasoning.
2.  **Google Custom Search API Key:** To use the Google Custom Search API.
3.  **Google Custom Search Engine ID (CSE ID):** For use with the Google Custom Search API.

**Obtaining API Keys:**

1.  **Gemini API Key:**
    *   Go to [Google AI Studio](https://makersuite.google.com/app/apikey).
    *   Create a new project or use an existing one.
    *   Create an API key.
2.  **Google Custom Search API Key and Search Engine ID:**
    *   Go to the [Google Cloud Console](https://console.cloud.google.com/).
    *   Create a new project or use an existing one.
    *   Enable the "Custom Search API" for your project.
    *   Go to "APIs & Services" -> "Credentials" and create an API key.
    *   Go to [Google Custom Search Engine](https://cse.google.com/cse/all).
    *   Create a new search engine and copy the "Search engine ID."

**Configuring Secrets:**

1.  Create a file named `.env` in the project's root directory.
2.  Add the following lines to the `.env` file, replacing the placeholder values with your actual API keys:

    ```
    GEMINI_API_KEY=your_gemini_api_key
    GOOGLE_API_KEY=your_google_api_key
    GOOGLE_CSE_ID=your_google_cse_id
    ```

## Installation and Execution

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/satyam9k/Hackathon_AI_News_Credibility.git
    cd AI-Powered-News-Credibility-Verification-System
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate the virtual environment
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```


