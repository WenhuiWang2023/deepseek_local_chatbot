# deepseek_local_chatbot
built a local chatbot with downloaded deepseek model
1. Install ollama  https://ollama.com
2. check the installation ollama --version
3. ollama serve if not working. netstat -ano | findstr :11434
4. ollama pull deepseek-r1:8b download deepseek r1 with 8b parameters version, check downloading status ollama list
5. setup environment: python -m venv deepseek_env  # Create a virtual environment
   source deepseek_env/Scripts/activate  # For Windows
   pip install -r requirements.txt
   pip install faiss-cpu
7. python admin.py
8. chainlit run app.py
