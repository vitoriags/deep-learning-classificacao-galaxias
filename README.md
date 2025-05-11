# Estudo Comparativo entre Arquiteturas de Redes Neurais Profundas para Classificação de Imagens de Galáxias

Este projeto apresenta um estudo comparativo entre diferentes arquiteturas de redes neurais profundas aplicadas à classificação morfológica de galáxias usando dados do projeto **Galaxy Zoo 2**.

Você pode ler o artigo completo em PDF: [Baixar artigo](https://github.com/vitoriags/deep-learning-classificacao-galaxias/raw/main/Estudo_Comparativo_entre_Arquiteturas_de_Redes_Neurais_Profundas_para_Classificação_de_Imagens_Baseado_em_Galáxias.pdf)

---

## Objetivo

Investigar o desempenho de arquiteturas modernas de **Deep Learning** para **classificar galáxias** em duas categorias principais: **Elípticas (E)** e **Espirais (S)**, com e sem pré-processamento de segmentação.

---

## Dataset

Utilizamos a base do **Galaxy Zoo 2**, composta por:

- **239.029 imagens rotuladas** (galáxias espirais e elípticas)
- Dados divididos em `70%` treino, `10%` validação e `20%` teste
- Versões: **imagens originais** e **imagens segmentadas** (com remoção de ruídos)

---

## Arquiteturas Avaliadas

### CNNs Básicas
- AlexNet
- MobileNetV2

### Redes Inception
- Inception V3
- Inception ResNet V2
- GoogLeNet (3 variações de saída)

### Redes ResNet
- ResNet152V2
- Attention ResNet 56
- Attention ResNet 92

### Transformers
- CoAtNet0 (convolução + Mecanismo de Atenção)

---

## Metodologia

- Treinamento com Keras
- Avaliação com **acurácia**, **precisão**, **recall**, **F1-score**
- Segmentação por componente conexa para reduzir ruído
- Teste das mesmas imagens em diferentes redes para comparação direta

---

## Resultados

| Arquitetura     | Tipo de Imagem | Acurácia (%) |
|-----------------|----------------|--------------|
| CoAtNet0        | Original       | 84.4         |
| CoAtNet0        | Segmentada     | 84.2         |
| GoogLeNet X1    | Original       | 84.1         |
| GoogLeNet X1    | Segmentada     | 84.0         |
| AlexNet         | Original       | 82.9         |
| AlexNet         | Segmentada     | 82.7         |
| ResNet152V2     | Segmentada     | 82.2         |

Redes com maior profundidade nem sempre superaram arquiteturas mais simples.

A **segmentação** teve impacto mínimo, sugerindo que as redes aprendem a focar nas regiões relevantes da imagem mesmo sem limpeza explícita.

---

## Conclusão

- Arquiteturas simples ainda são eficazes dependendo da tarefa.
- Redes como CoAtNet mostram grande potencial quando há dados suficientes.
- Esse projeto reforça o papel do **Deep Learning em automação científica**, especialmente em problemas de **visão computacional**.

---

## Autores

**Allan Maffra, Vitória Gabriely Sousa e Caio Rafael do Nascimento Santiago**

---
