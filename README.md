<img width="1108" height="468" alt="image" src="https://github.com/user-attachments/assets/dde69f4f-5282-4a1a-8732-c9c456f8e0dc" />

# MLOps Churn Prediction Project

## Objetivo

Desenvolver um pipeline completo de Machine Learning com foco em MLOps para previsão de churn de clientes, simulando um cenário real de tomada de decisão com impacto financeiro.

---

## Problema de Negócio

Empresas perdem receita devido à incapacidade de identificar clientes com risco de cancelamento antecipadamente.

Este projeto busca prever churn e permitir ações proativas de retenção.

---

## Abordagem

- Dataset sintético com comportamento probabilístico (simulando cenário real)
- Modelagem com múltiplos algoritmos
- Ajuste de threshold para controle de trade-offs
- Seleção automática do melhor modelo baseada em impacto financeiro
- Versionamento e governança com MLflow
- Deploy via API com FastAPI

---

## Arquitetura

```Dados → Treinamento → MLflow → Model Registry → API → Consumo (CRM / Sistemas) ```Lucro = (clientes retidos * receita) - custo de ação


---

## Modelos Utilizados

- Random Forest
- Logistic Regression

Utilização de `predict_proba` para permitir ajuste de threshold e tomada de decisão mais flexível.

---

## Métricas

- Accuracy
- Precision
- Recall
- F1-score

Foco em recall para maximizar a identificação de clientes com risco de churn.

---

## Diferencial (Business Driven ML)

A escolha do modelo não é baseada apenas em métricas tradicionais.

Foi implementada uma função de negócio:

```Lucro = (clientes retidos * receita) - custo de ação```


O melhor modelo é selecionado com base no maior impacto financeiro.

---

## MLOps

Uso do MLflow para:

- Tracking de experimentos
- Registro de métricas e parâmetros
- Versionamento de modelos
- Model Registry

---

## Deploy

O modelo é disponibilizado via API utilizando FastAPI:

POST /predict


Exemplo de entrada:

```json
{
  "tenure": 10, --Tempo de vigência
  "MonthlyCharges": 80, --Preços mensais
  "TotalCharges": 500, --Preços totais
  "Contract": "Month-to-month", --Tipo de Contrato
  "InternetService": "Fiber optic" --Serviço de internet
}
```

Resultados
Pipeline completo de MLOps
Comparação automática de modelos
Otimização baseada em impacto financeiro
Versionamento e controle de modelos
API funcional para consumo em tempo real

Próximos Passos
Monitoramento de modelo (data drift / performance)
Integração com cloud (Azure / Databricks)
Pipeline CI/CD automatizado
Uso de dados reais

Conclusão

Mais do que construir modelos, o objetivo é transformar dados em decisões confiáveis, com versionamento, rastreabilidade e foco em impacto real no negócio, conectando Machine Learning, engenharia e governança para aproximar a solução de um ambiente de produção.


