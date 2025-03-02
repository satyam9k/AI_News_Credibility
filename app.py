########################################
#Author- Satyam Kumar
########################################
#import libraries - make sure to run requirements.txt before
import streamlit as st
import google.generativeai as genai
import requests
import json
import easyocr
from PIL import Image
import pandas as pd
from langdetect import detect
from translate import Translator
import re
from urllib.parse import quote
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

#environment variables from .env file
load_dotenv()

# API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  

# configure the app for streamlit
st.set_page_config(
    page_title="News Credibility Verifier",
    page_icon="📰",
    layout="wide"
)

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("GEMINI_API_KEY not found in .env file.")

#gemini model
model = genai.GenerativeModel('gemini-2.0-flash')
vision_model = genai.GenerativeModel('gemini-2.0-flash')

#trusted sources
TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "apnews.com", "npr.org", "theguardian.com",
    "nytimes.com", "wsj.com", "washingtonpost.com", "economist.com",
    "cnn.com", "time.com", "politico.com", "factcheck.org", "snopes.com",
    "thehindu.com", "indianexpress.com", "ndtv.com",  "hindustantimes.com",  
    "timesofindia.indiatimes.com", "livemint.com" 
]


#easyocr for extracting text from image
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'])

# OCR 
def extract_text_from_image(image):
    try:
        if isinstance(image, Image.Image):
            import numpy as np
            image_np = np.array(image)
        else:
            image_np = image
        reader = load_ocr_reader()
        result = reader.readtext(image_np)
        extracted_text = " ".join([text[1] for text in result])
        return extracted_text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

# language detection and translation
def process_language(text):
    try:
        detected_lang = detect(text)
        original_language = detected_lang
        #translate if not in English
        if detected_lang != 'en':
            translator = Translator(to_lang="en")
            text = translator.translate(text)

        return text, original_language
    except Exception as e:
        st.error(f"Language processing error: {e}")
        return text, "unknown"


#using Custom Search JSON API
def search_web(query, num_results=5):
    try:
        #Gemini to generate search queries
        search_prompt = f"""
        Act as a highly sophisticated AI agent specialized in information retrieval and news verification.
        Your task is to research the following news content and find relevant, credible sources:
        
        News Content: {query}
        
        Instructions:
        1.  Based on the news content, identify 3-4 specific, insightful search queries that would help verify or refute the information.
           - Ensure that the queries are specific and cover different aspects of the news content.
        2. Generate search queries that if used on a search engine like Google, would get information about the news content.
        3. Do not search for the results. Return ONLY the search queries, one per line.
           - Do not include any title, number or any other character other than the search query.
           - Do not include any introduction or conclusion.
           - Do not include any explanation.
           - Return only one search query per line
           - All the search queries should be on english.
        """

        search_response = model.generate_content(search_prompt)
        search_queries = [q.strip() for q in search_response.text.strip().split('\n') if q.strip()]

        all_results = []
        for search_query in search_queries:
            if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
                st.error("Google API Key and Google CSE ID are required for web search. Please check your .env file.")
                return []

            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            res = (
                service.cse()
                .list(q=search_query, cx=GOOGLE_CSE_ID, num=2)
                .execute()
            )
            
            if "items" in res:
                for item in res["items"]:
                    all_results.append({
                        "title": item["title"],
                        "source": item["link"].split('/')[2] if len(item["link"].split('/')) >= 3 else "N/A",
                        "snippet": item["snippet"],
                        "url": item["link"]
                    })

        return all_results[:6]  # limit to top 6 results (to reduce the chnace of api limit exhaustion)
    except Exception as e:
        st.error(f"Web search error: {e}")
        return []

# RAG( Retrival augmented generation)
def enhance_with_rag(article_text, search_results):
    try:
        context = "\n\n".join([f"Source: {result['source']}\nTitle: {result['title']}\nExcerpt: {result['snippet']}"
                               for result in search_results])

        rag_prompt = f"""
        You are an expert news fact-checker.
        I have a news article and several reference sources. Use the reference information to verify the claims in the article.

        ARTICLE:
        {article_text}

        REFERENCE SOURCES:
        {context}

        Instructions:
        1. Analyze how well the article is supported by the reference sources.
        2. Be cautious. If there are slight differences in wording or minor facts, do not immediately label them as contradictions.
           - Consider that different sources may report the same event in slightly different ways.
        3. Identify key claims in the article and determine whether they are confirmed, contradicted, or unsupported.
           - A confirmed claim should have clear, direct support from the references.
           - A contradicted claim should have clear evidence that it's incorrect.
           - An unsupported claim is one that's mentioned in the article but not mentioned in any reference.
        4. Provide an overall assessment. Consider the proportion of confirmed, contradicted, and unsupported claims.

        Return your analysis in JSON format:
        {{
            "supported_claims": ["claim 1", "claim 2"...],
            "contradicted_claims": ["claim 1", "claim 2"...],
            "unsupported_claims": ["claim 1", "claim 2"...],
            "overall_assessment": "brief summary of verification"
        }}
        """

        rag_response = model.generate_content(rag_prompt)

        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', rag_response.text)
            if json_match:
                rag_analysis = json.loads(json_match.group(1))
            else:
                rag_analysis = json.loads(rag_response.text)
        except json.JSONDecodeError:
            st.warning("Structured analysis format unavailable. Using raw analysis.")
            rag_analysis = {
                "supported_claims": [],
                "contradicted_claims": [],
                "unsupported_claims": [],
                "overall_assessment": rag_response.text
            }

        return rag_analysis
    except Exception as e:
        st.error(f"RAG processing error: {e}")
        return {
            "supported_claims": [],
            "contradicted_claims": [],
            "unsupported_claims": [],
            "overall_assessment": "Error in processing with RAG"
        }

#credibility score
def calculate_credibility_score(article_text, search_results, rag_analysis):
    source_score = 0
    content_similarity_score = 0
    contradiction_penalty = 0
    unsupported_penalty = 0
    supported_bonus = 0  

    trusted_count = 0
    for result in search_results:
        domain = result.get('source', '').lower()
        if any(trusted in domain for trusted in TRUSTED_SOURCES):
            trusted_count += 1

    source_score = min(trusted_count / max(1, len(search_results)) * 40, 40)

    # content similarity
    similarity_prompt = f"""
    You are a expert.
    Compare the following news article with these reference snippets.
    Provide a brief assessment of the overall factual similarity between the article and the references.
    Evaluate if the main facts and claims in the article are generally consistent with the references.

    ARTICLE:
    {article_text}

    REFERENCE SNIPPETS:
    {[result.get('snippet', '') for result in search_results]}
    
    Possible options to return : "Very similar" , "Similar" , "Mixed" , "Not similar"

    Return ONLY one of these options without any explanation.
    """

    try:
        similarity_response = model.generate_content(similarity_prompt)
        similarity_text = similarity_response.text.strip()

        if "Very similar" in similarity_text:
            content_similarity_score = 40
        elif "Similar" in similarity_text:
            content_similarity_score = 30
        elif "Mixed" in similarity_text:
            content_similarity_score = 15
        else:
            content_similarity_score = 0

    except:
        content_similarity_score = 0

    # supported claims bonus
    supported_claims_count = len(rag_analysis.get('supported_claims', []))
    supported_bonus = supported_claims_count * 10  

    #check if source is from a trusted site for each supported claim.
    for claim in rag_analysis.get("supported_claims", []):
      for result in search_results:
        domain = result.get("source", "").lower()
        if any(trusted in domain for trusted in TRUSTED_SOURCES) and claim.lower() in result.get('snippet', '').lower():
          supported_bonus += 5

    contradiction_penalty = len(rag_analysis.get('contradicted_claims', [])) * 5  
    unsupported_penalty = len(rag_analysis.get("unsupported_claims", [])) * 3 

    raw_score = source_score + content_similarity_score + supported_bonus - contradiction_penalty - unsupported_penalty
    final_score = max(min(raw_score, 100), 0)

    return round(final_score, 1)


# main verification function
def verify_news(article_text, image=None):
    if image is not None:
        image_text = extract_text_from_image(image)
        if image_text:
            article_text += "\n\nExtracted from image: " + image_text

    processed_text, original_language = process_language(article_text)

    search_results = search_web(processed_text)

    rag_analysis = enhance_with_rag(processed_text, search_results)

    credibility_score = calculate_credibility_score(processed_text, search_results, rag_analysis)

    return {
        "original_language": original_language,
        "processed_text": processed_text,
        "search_results": search_results,
        "rag_analysis": rag_analysis,
        "credibility_score": credibility_score
    }

# streamlit UI
def main():
    st.title("🔍 AI-Powered News Credibility Verification System")
    st.markdown("""
    This tool helps verify the credibility of news articles by cross-checking them against trusted sources.
    Simply paste a news article or upload an image containing news content to get started.
    """)

    input_method = st.radio("Select input method:", ["Text", "Image", "Both"])

    article_text = ""
    uploaded_image = None

    if input_method in ["Text", "Both"]:
        article_text = st.text_area("Enter news article text:", height=200)

    if input_method in ["Image", "Both"]:
        uploaded_image = st.file_uploader("Upload an image containing news content:", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    with st.expander("Advanced Options"):
        st.write("Custom Trusted Sources:")
        custom_sources = st.text_area("Enter custom trusted sources (one per line):", value="\n".join(TRUSTED_SOURCES))
        custom_trusted_sources = [src.strip() for src in custom_sources.split("\n") if src.strip()]
    

    if st.button("Verify News"):
        if not article_text and uploaded_image is None:
            st.error("Please provide either text or an image to verify.")
            return

        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY is missing. Please add it to the .env file")
            return

        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            st.error("GOOGLE_API_KEY or GOOGLE_CSE_ID is missing. Please add it to the .env file.")
            return

        with st.spinner("Verifying news content... This may take a moment."):
            image = None
            if uploaded_image is not None:
                image = Image.open(uploaded_image)

            verification_result = verify_news(article_text, image)

            st.header("Verification Results")

            score = verification_result["credibility_score"]
            if score >= 70:
                st.markdown(f"### Credibility Score: <span style='color:green; font-size:24px;'>{score}/100</span> ✅", unsafe_allow_html=True)
                st.success("This article appears to be credible.")
            elif score >= 40:
                st.markdown(f"### Credibility Score: <span style='color:orange; font-size:24px;'>{score}/100</span> ⚠️", unsafe_allow_html=True)
                st.warning("This article has mixed credibility. Check the details below.")
            else:
                st.markdown(f"### Credibility Score: <span style='color:red; font-size:24px;'>{score}/100</span> ❌", unsafe_allow_html=True)
                st.error("This article has low credibility. Exercise caution.")

            if verification_result["original_language"] != "en":
                st.info(f"Original language detected: {verification_result['original_language']}. The content was translated for verification.")

            st.subheader("Content Analysis")
            rag_analysis = verification_result["rag_analysis"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Supported Claims:")
                if rag_analysis.get("supported_claims"):
                    for claim in rag_analysis["supported_claims"]:
                        st.markdown(f"- ✅ {claim}")
                else:
                    st.write("No supported claims identified.")

            with col2:
                st.markdown("##### Contradicted Claims:")
                if rag_analysis.get("contradicted_claims"):
                    for claim in rag_analysis["contradicted_claims"]:
                        st.markdown(f"- ❌ {claim}")
                else:
                    st.write("No contradicted claims identified.")

            st.markdown("##### Unsupported Claims:")
            if rag_analysis.get("unsupported_claims"):
                for claim in rag_analysis["unsupported_claims"]:
                    st.markdown(f"- ❓{claim}")
            else:
                st.write("No unsupported claims identified.")

            st.markdown("##### Overall Assessment:")
            st.write(rag_analysis.get("overall_assessment", "No overall assessment available."))

            st.subheader("Reference Sources")
            if verification_result["search_results"]:
                for result in verification_result["search_results"]:
                    st.markdown(f"**[{result['title']}]({result['url']})**")
                    st.write(f"Source: {result['source']}")
                    st.write(f"Snippet: {result['snippet']}")
                    st.write("---")
            else:
                st.write("No reference sources found.")

if __name__ == "__main__":
    main()
