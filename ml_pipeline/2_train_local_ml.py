import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # 1. Загрузка данных
    try:
        df = pd.read_csv('flux_dataset.csv')
    except FileNotFoundError:
        print("Ошибка: Файл 'flux_dataset.csv' не найден. Сначала запустите скрипт генерации датасета.")
        return

    # 2. Предобработка
    print(f"Загружено строк: {len(df)}")
    df = df.dropna(subset=['code', 'label'])
    print(f"Строк после очистки: {len(df)}")

    # 3. Разделение на фичи и таргет
    X = df['code']
    y = df['label']

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Создание пайплайна
    # TfidfVectorizer настроен на n-граммы (1, 3) для захвата синтаксических конструкций Python
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3), 
            max_features=3000,
            analyzer='char_wb' # Анализ по символьным n-граммам внутри слов (лучше для кода)
        )),
        ('clf', RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 6. Обучение
    print("Начинаю обучение модели...")
    pipeline.fit(X_train, y_train)
    print("Обучение завершено.")

    # 7. Оценка
    y_pred = pipeline.predict(X_test)
    
    print("\n=== Метрики качества ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Сохранение модели
    model_filename = 'flux_local_model.pkl'
    joblib.dump(pipeline, model_filename)
    print(f"\nМодель сохранена в файл: {model_filename}")

if __name__ == "__main__":
    train_model()