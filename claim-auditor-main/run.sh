conda env create -f environment.yml
conda activate claim-auditor
cp .env.example .env   # fill in your keys

# terminal 1
uvicorn api.main:app --reload --port 8000

# terminal 2
streamlit run ui/app.py