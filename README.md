# RestAPI-ML

Uma implementação de referência profissional de uma API REST de Machine Learning usando **FastAPI**, **Scikit-Learn** e **Docker**.

## 🚀 Funcionalidades

- **FastAPI**: Framework de alta performance e fácil aprendizado.
- **Machine Learning**: Classificador Random Forest treinado no dataset Iris.
- **Pronto para Produção**:
    - **Pydantic Settings**: Configuração tipada e segura.
    - **Logging**: Configuração de logs estruturados.
    - **Docker**: Containerizado para fácil implantação.
    - **Arquitetura**: Separação limpa de responsabilidades (Rotas, Serviços, Schemas).

## 🛠️ Estrutura do Projeto

```
RestAPI-ML/
├── saved_models/       # Onde os modelos treinados são armazenados
├── scripts/            # Scripts auxiliares (ex: treinamento)
├── src/
│   ├── routes/         # Endpoints da API
│   ├── schemas/        # Modelos Pydantic (Entrada/Saída)
│   ├── services/       # Lógica de Negócio e Inferência do Modelo
│   ├── utils/          # Logger, Exceções
│   └── config.py       # Configuração
├── main.py             # Ponto de Entrada da Aplicação
├── Dockerfile          # Instruções de Build do Docker
├── docker-compose.yml  # Orquestração de Containers
└── requirements.txt    # Dependências Python
```

## 🏃 Executando Localmente

1. **Instalar Dependências**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Treinar o Modelo**:
   Isso gera o arquivo `saved_models/iris_model.joblib`.
   ```bash
   python scripts/train_model.py
   ```

3. **Iniciar a API**:
   ```bash
   uvicorn main:app --reload
   ```

4. **Acessar Documentação**:
   Abra o navegador em [http://localhost:8000/docs](http://localhost:8000/docs) para ver a interface interativa do Swagger.

## 🐳 Executando com Docker

1. **Construir e Rodar**:
   ```bash
   docker-compose up --build
   ```

2. **Acesso**:
   A API estará disponível em [http://localhost:8000](http://localhost:8000).

## 🧪 Endpoints da API

### `POST /api/v1/predict`
Exemplo de uso para previsão:

**Corpo da Requisição:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Resposta:**
```json
{
  "class_id": 0,
  "class_name": "setosa"
}
```

### `GET /api/v1/health`
Verifica se o serviço está rodando e se o modelo foi carregado corretamente.