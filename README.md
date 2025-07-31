# st-sabaling

## Make a Virtual Environment (Recommended)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Make a .env file
```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

## Add the folder data/ to insert the pdfs
```
data
└───yourpdf.pdf
```

## Running the Streamlit
```
streamlit run app.py
```
