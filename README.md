# 🃏 AI Poker - Sistema de Classificação de Mãos de Poker com IA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Um sistema completo de Inteligência Artificial para classificação automática de mãos de poker utilizando algoritmos de Machine Learning. O projeto implementa um pipeline end-to-end desde o processamento de dados até o treinamento e avaliação de modelos.

## 🎯 Visão Geral

Este projeto desenvolve um sistema de IA capaz de:

- **Classificar automaticamente** diferentes tipos de mãos de poker
- **Processar datasets** com mais de 1 milhão de registros
- **Treinar modelos** usando algoritmos de clustering (K-Means)
- **Avaliar performance** com métricas abrangentes
- **Fornecer API** para predições em tempo real

## 🚀 Características Principais

- ✅ **Pipeline Completo**: Conversão, treinamento, avaliação e persistência
- ✅ **Interface CLI**: Comandos simples para todas as operações
- ✅ **API de Predição**: Servidor web para uso em produção
- ✅ **Múltiplos Formatos**: Suporte a dados textuais e numéricos
- ✅ **Avaliação Robusta**: Métricas de clustering e acurácia
- ✅ **Documentação Completa**: Guias e exemplos detalhados

## 📊 Tipos de Mão Classificados

| Código | Tipo de Mão     | Descrição                                |
| ------ | --------------- | ---------------------------------------- |
| 0      | Nothing         | Sem combinação                           |
| 1      | One Pair        | Um par                                   |
| 2      | Two Pair        | Dois pares                               |
| 3      | Three of a Kind | Trinca                                   |
| 4      | Straight        | Sequência                                |
| 5      | Flush           | Mesmo naipe                              |
| 6      | Full House      | Trinca + Par                             |
| 7      | Four of a Kind  | Quadra                                   |
| 8      | Straight Flush  | Sequência + Mesmo naipe                  |
| 9      | Royal Flush     | Ás, Rei, Dama, Valete, 10 do mesmo naipe |

## 🏗️ Arquitetura do Sistema

```
AI_Poker/
├── poker_ai/                    # Módulo principal
│   ├── cluster.py              # Algoritmo K-Means
│   ├── data.py                 # Processamento de dados
│   ├── cards.py                # Conversão de símbolos
│   ├── cli.py                  # Interface de linha de comando
│   └── requirements.txt        # Dependências
├── resultados/                  # Modelos e datasets
│   ├── modelo_100_porcento.pkl
│   ├── modelo_split_70_30.pkl
│   └── *.csv                   # Datasets processados
├── Scripts/                     # Scripts auxiliares
├── Documentos/                  # Documentação e relatórios
└── Scripts principais
    ├── treinar_completo.py     # Pipeline automático
    ├── ver_resultados.py      # Verificação de modelos
    ├── api_predicao.py         # API web
    └── guia_de_uso.py          # Guia detalhado
```

## 🛠️ Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação das Dependências

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/AI_Poker.git
cd AI_Poker

# Instale as dependências
pip install -r poker_ai/requirements.txt

# Ou instale manualmente
pip install pandas scikit-learn numpy
```

## 🚀 Uso Rápido

### 1. Pipeline Completo (Recomendado)

```bash
# Executa todo o pipeline automaticamente
python treinar_completo.py

# Ou com limite de dados para teste
python treinar_completo.py --max-linhas 10000
```

### 2. Treinamento Manual

```bash
# Treinar com todos os dados
python -m poker_ai dados.csv full --label-column tipo_mao --model-output modelo.pkl

# Treinar com split 70/30
python -m poker_ai dados.csv split --label-column tipo_mao --train-size 0.7 --model-output modelo.pkl
```

### 3. Verificar Resultados

```bash
# Verificar métricas dos modelos treinados
python ver_resultados.py
```

### 4. Usar API de Predição

```bash
# Iniciar servidor web
python api_predicao.py

# Acessar: http://localhost:5000
```

## 📋 Formato dos Dados

### Estrutura do CSV

```csv
carta1_naipe,carta1_valor,carta2_naipe,carta2_valor,carta3_naipe,carta3_valor,carta4_naipe,carta4_valor,carta5_naipe,carta5_valor,tipo_mao
0,2,0,3,0,4,0,5,0,6,8
1,10,1,11,1,12,1,13,1,1,9
```

### Mapeamento de Valores

- **Naipes**: 0=HEARTS, 1=DIAMONDS, 2=CLUBS, 3=SPADES
- **Valores**: 1=Ás, 2-10=Números, 11=Valete, 12=Dama, 13=Rei
- **Tipo de Mão**: 0-9 (ver tabela acima)

## 📊 Resultados de Performance

### Métricas Obtidas

| Modelo         | Inércia     | Silhouette Score | Acurácia    |
| -------------- | ----------- | ---------------- | ----------- |
| 100% dos dados | 113,872.69  | 0.141            | 44.8%       |
| Split 70/30    | [Calculado] | [Calculado]      | [Calculado] |

### Interpretação

- **Inércia**: Quanto menor, melhor a compactação dos clusters
- **Silhouette Score**: 0.141 indica clusters parcialmente separados
- **Acurácia**: 44.8% é aceitável para clustering não-supervisionado

## 🔧 Scripts Disponíveis

### Scripts Principais

- **`treinar_completo.py`**: Pipeline automático completo
- **`ver_resultados.py`**: Verificação e métricas dos modelos
- **`api_predicao.py`**: Servidor web para predições
- **`guia_de_uso.py`**: Guia detalhado de uso

### Scripts de Processamento

- **`converter_dataset.py`**: Conversão de formatos de dados
- **`preparar_dataset.py`**: Validação e limpeza de datasets
- **`test_modelo_completo.py`**: Testes automatizados

## 📚 Documentação

### Guias Disponíveis

- **`INSTRUCOES_RAPIDAS.txt`**: Comandos essenciais
- **`RELATORIO_IA_POKER.md`**: Relatório técnico completo
- **`RELATORIO_DETALHADO.txt`**: Análise detalhada dos resultados

### Exemplos de Uso

```python
from poker_ai.cluster import PokerHandClusterer

# Carregar modelo
modelo = PokerHandClusterer.load_model('resultados/modelo_split_70_30.pkl')

# Fazer predição
mao = np.array([[0, 2, 0, 3, 0, 4, 0, 5, 0, 6]])  # 5 cartas
cluster = modelo.predict(mao)
tipo_mao = modelo.predict_labels(mao)
```

## 🧪 Testes

```bash
# Executar testes completos
python test_modelo_completo.py

# Validar dataset
python preparar_dataset.py --arquivo dados.csv --validar --estatisticas

# Limpar dataset
python preparar_dataset.py --arquivo dados.csv --limpar --saida dados_limpo.csv
```

## 🔍 Comandos Úteis

```bash
# Ajuda geral
python -m poker_ai --help

# Ajuda do preparador
python preparar_dataset.py --help

# Criar dataset de exemplo
python preparar_dataset.py --criar-exemplo --n-amostras 1000

# Verificar resultados
python ver_resultados.py
```

## 🎯 Casos de Uso

### 1. Desenvolvimento de Jogos

- Classificação automática de mãos em jogos de poker
- Validação de regras de poker
- Sistema de pontuação inteligente

### 2. Análise de Dados

- Estudos estatísticos de poker
- Análise de padrões em jogadas
- Pesquisa acadêmica em jogos

### 3. Aplicações Educacionais

- Ferramenta de aprendizado de poker
- Sistema de tutoria inteligente
- Plataforma de treinamento

## 🚧 Limitações e Melhorias Futuras

### Limitações Atuais

- ⚠️ Acurácia de 44.8% pode ser melhorada
- ⚠️ Silhouette Score baixo (0.141)
- ⚠️ Algoritmo K-Means pode não ser ideal para dados complexos

### Melhorias Planejadas

- 🔧 Implementar algoritmos supervisionados (Random Forest, SVM)
- 🔧 Adicionar feature engineering avançado
- 🔧 Usar ensemble methods para melhorar performance
- 🔧 Aplicar técnicas de balanceamento de classes
- 🔧 Validar com datasets maiores

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Autor

**Seu Nome**

- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [Seu Perfil](https://linkedin.com/in/seu-perfil)

## 🙏 Agradecimentos

- [scikit-learn](https://scikit-learn.org/) pela biblioteca de Machine Learning
- [pandas](https://pandas.pydata.org/) pela manipulação de dados
- [numpy](https://numpy.org/) pela computação numérica
- Comunidade Python pelo suporte e recursos

## 📞 Suporte

Se você encontrar algum problema ou tiver dúvidas:

- 📧 Email: seu-email@exemplo.com
- 🐛 Issues: [GitHub Issues](https://github.com/seu-usuario/AI_Poker/issues)
- 💬 Discussões: [GitHub Discussions](https://github.com/seu-usuario/AI_Poker/discussions)

---

**Desenvolvido com ❤️ para aprendizado de Machine Learning e Poker! 🃏✨**

_Se este projeto foi útil para você, considere dar uma ⭐ no GitHub!_
