from joblib import load
import os

def load_model(model_path: str = None):
    """
    Carrega o modelo treinado a partir do caminho especificado.
    Se não for informado, usa o modelo salvo em artifacts/best_model.pkl.
    """
    if model_path is None:
        # Assume que a pasta artifacts está na raiz do repositório
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "artifacts", "best_model_1.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
    return load(model_path)
