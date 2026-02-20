import os
import time
import json
import pandas as pd
import google.generativeai as genai
from typing import List, Dict

# Настройка API ключа
# Рекомендуется использовать переменные окружения для безопасности
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=API_KEY)

def fetch_batch(batch_size: int = 10) -> List[Dict]:
    """
    Отправляет промпт в Gemini для генерации батча примеров кода.
    """
    # Используем актуальную модель gemini-3-flash-preview
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    prompt = f"""
    Generate a synthetic dataset for training a machine learning model to detect malicious Python code.
    Provide exactly {batch_size} examples of 'clean' code and {batch_size} examples of 'malicious' code.
    
    Clean code should include:
    - Mathematical calculations
    - Class definitions
    - Standard algorithms (sorting, searching)
    - Data processing with pandas/numpy
    
    Malicious code (malware/backdoors) should include:
    - Use of eval() or exec() for dynamic execution
    - Base64 obfuscation of commands
    - Socket-based reverse shells
    - subprocess.Popen for system manipulation
    - Unauthorized file access (e.g., reading /etc/passwd or sensitive files)
    
    STRICT REQUIREMENT: Return the response ONLY as a valid JSON array of objects.
    Each object must have two keys: "code" (the Python snippet) and "label" (0 for clean, 1 for malicious).
    
    Format example:
    [
        {{"code": "print('hello')", "label": 0}},
        {{"code": "import os; os.system('rm -rf /')", "label": 1}}
    ]
    """
    
    try:
        response = model.generate_content(prompt)
        # Очистка ответа от markdown-разметки (если Gemini добавит ```json)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        batch_data = json.loads(text)
        return batch_data
    except Exception as e:
        print(f"Error during Gemini request or JSON parsing: {e}")
        return []

def main():
    dataset = []
    iterations = 5
    batch_size = 10 # 10 clean + 10 malicious per batch = 20 total per iteration
    
    print(f"Starting dataset generation ({iterations} iterations)...")
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        batch = fetch_batch(batch_size)
        
        if batch:
            dataset.extend(batch)
            print(f"Successfully added {len(batch)} examples.")
        else:
            print("Batch failed, skipping...")
            
        if i < iterations - 1:
            print("Waiting 4 seconds to respect rate limits...")
            time.sleep(4)
            
    if dataset:
        # Сохранение в CSV
        df = pd.DataFrame(dataset)
        output_file = "flux_dataset.csv"
        df.to_csv(output_file, index=False)
        print(f"Generation complete! Total examples: {len(dataset)}")
        print(f"Dataset saved to {output_file}")
    else:
        print("No data generated.")

if __name__ == "__main__":
    main()