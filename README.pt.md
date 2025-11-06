# 🎥 Video Analysis Tech Challenge

Bem-vindo ao projeto **Video Analysis**! Este repositório contém a solução para o desafio proposto na disciplina, cujo objetivo é implementar uma aplicação capaz de realizar análise de vídeo com técnicas de visão computacional e aprendizado de máquina.

---

## 📝 Descrição do Desafio

O projeto aborda o seguinte problema:

> **O PROBLEMA:**  
> O Tech Challenge desta fase consiste na criação de uma aplicação que utilize análise de vídeo.  
> O projeto deve incorporar técnicas de reconhecimento facial, análise de expressões emocionais em vídeos e detecção de atividades.

### 📋 Funcionalidades Requeridas

1. **Reconhecimento facial**: Identifique e marque os rostos presentes no vídeo.
2. **Análise de expressões emocionais**: Analise as expressões emocionais dos rostos identificados.
3. **Detecção de atividades**: Detecte e categorize as atividades sendo realizadas no vídeo.
4. **Geração de resumo**: Crie um resumo automático das principais atividades e emoções detectadas no vídeo.

---

## 🚀 Tecnologias Utilizadas

- Python
- OpenCV
- Deep Learning (Ex: TensorFlow, Keras ou PyTorch)
- Modelos pré-treinados para detecção facial e análise de emoções
- Outras bibliotecas auxiliares para manipulação de vídeo e visualização

---

## ⚙️ Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/guillherms/video-analysis.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Coloque o vídeo fornecido na pasta `input/` (crie se necessário).
4. Execute o script principal:
   ```bash
   python main.py
   ```
5. Os resultados serão salvos na pasta `output/`.

---

## 📑 Estrutura do Projeto

```
video-analysis/
├── input/                 # Vídeos de entrada
├── output/                # Resultados e resumos gerados
├── src/                   # Código-fonte principal
│   ├── facial_recognition.py
│   ├── emotion_analysis.py
│   └── activity_detection.py
├── requirements.txt
└── main.py
```

---

## 🛠️ Histórico de Desenvolvimento

| Data       | Alteração                                                                                   |
|------------|--------------------------------------------------------------------------------------------|
| 2024-06-01 | 🎉 Início do projeto e definição da proposta                                                |
| 2024-06-02 | 👤 Implementação inicial do reconhecimento facial                                           |
| 2024-06-03 | 😊 Adicionada análise de expressões emocionais nos rostos detectados                       |
| 2024-06-04 | 🏃‍♂️ Implementação da detecção e categorização de atividades no vídeo                      |
| 2024-06-05 | 📝 Geração automática de resumo das atividades e emoções detectadas                         |
| 2024-06-06 | 🐞 Correções de bugs e melhorias na interface de visualização dos resultados                |
| 2024-06-07 | 📦 Refatoração do código e atualização da documentação                                      |
| ...        | ...                                                                                        |

---

## 💡 Possíveis Melhorias Futuras

- Suporte a múltiplos vídeos simultaneamente
- Interface gráfica para facilitar uso
- Exportação dos resultados em diferentes formatos (PDF, Excel)
- Integração com APIs de terceiros para análise avançada

---

## 📄 Licença

Este projeto é apenas para fins acadêmicos.

---

## 🙋‍♂️ Contato

Dúvidas ou sugestões?  
Entre em contato via [GitHub](https://github.com/guillherms) ou [LinkedIn](https://www.linkedin.com/in/guilherme-santos-de-oliveira-ba9986161/)

---