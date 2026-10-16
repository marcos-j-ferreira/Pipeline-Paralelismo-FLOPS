# Pipeline Parallelism — Treinamento Orientado por Orçamento Computacional (FLOPs)

Experimento de treinamento de modelos de linguagem com orçamento computacional definido em FLOPs, utilizando **pipeline parallelism entre GPUs NVIDIA T4**. O objetivo é estimar o tamanho de modelo treinável dentro de um orçamento computacional limitado, validar a quantidade de tokens processados e avaliar os resultados obtidos com o dataset TinyStories.

## 1\. Objetivo do experimento

O primeiro experimento teve como objetivo validar uma estimativa teórica de treinamento baseada no orçamento computacional disponível, comparando-a com o desempenho observado na prática.

A proposta foi treinar um modelo de linguagem de aproximadamente 11,4 milhões de parâmetros durante cerca de 20 minutos, utilizando duas GPUs NVIDIA T4 e uma estratégia de paralelismo em pipeline.

Para estimar o tamanho do modelo e a quantidade de tokens necessários, foi utilizada uma aproximação baseada na relação entre orçamento computacional, número de parâmetros e quantidade de tokens de treinamento.

## 2\. Estimativa do orçamento computacional

O orçamento computacional foi estimado a partir da seguinte expressão:

$$
C = 65 \times 10^{12} \times 0{,}1 \times 1200 \times 2
$$

$$
C = 1{,}56 \times 10^{16}\ \text{FLOPs}
$$

Nessa estimativa, foram considerados:

- **65 TFLOPs:** capacidade de referência utilizada no cálculo.
- **0,1:** fator de utilização computacional estimado, equivalente a 10%.
- **1.200 segundos:** duração planejada do treinamento, correspondente a 20 minutos.
- **2:** número de GPUs utilizadas.

O valor de 65 TFLOPs representa uma referência teórica de desempenho, não necessariamente o desempenho efetivo de cada GPU durante o treinamento. Portanto, o orçamento calculado deve ser interpretado como uma estimativa dependente das premissas adotadas.

### 2.1. Estimativa do número de parâmetros

Para estimar o tamanho do modelo, foi utilizada a relação aproximada:

$$
C \approx 120ND
$$

Considerando a proporção de 20 tokens por parâmetro, temos:

$$
D = 20N
$$

Substituindo na equação anterior:

$$
C \approx 120N(20N) = 2400N^2
$$

Assim, o número estimado de parâmetros é:

$$
N \approx \sqrt{\frac{C}{2400}}
$$

$$
N \approx \sqrt{\frac{1{,}56 \times 10^{16}}{2400}}
$$

$$
N \approx 80{,}6\ \text{milhões de parâmetros}
$$

**Observação:** essa estimativa depende da convenção de FLOPs adotada. A fórmula inicial `N = √(C/120)` não é compatível com a hipótese `D = 20N`; para manter essa hipótese, o denominador correto é 2.400.

Para um modelo de 11,4 milhões de parâmetros, a estimativa correspondente seria:

$$
C \approx 2400(11{,}4 \times 10^6)^2
$$

$$
C \approx 3{,}12 \times 10^{17}\ \text{FLOPs}
$$

Portanto, o orçamento de $1{,}56 \times 10^{16}$ FLOPs, sob essa convenção, não é suficiente para justificar teoricamente um modelo de 11,4 milhões de parâmetros treinado com 20 tokens por parâmetro.

## 3\. Execução e quantidade de tokens

O treinamento foi executado inicialmente durante aproximadamente 20 minutos, utilizando duas GPUs T4.

| Métrica | Resultado |
| --- | --- |
| GPUs utilizadas | 2 × NVIDIA T4 |
| Tempo inicial planejado | 20 minutos |
| Tokens previstos no planejamento original | 228.000.000 |
| Tokens processados inicialmente | 218.116.800 |
| Tokens após o aumento do orçamento | 229.068.000 |

Na execução inicial, o modelo processou 218.116.800 tokens, ficando abaixo da meta original de 228 milhões.

Para compensar essa diferença, o orçamento de treinamento foi ampliado em aproximadamente 5%. Com o orçamento adicional, o modelo processou 229.068.000 tokens, superando a meta inicial em cerca de 1,07 milhão de tokens.

Esse resultado demonstra que o aumento do orçamento permitiu atingir a quantidade de tokens planejada. Entretanto, o número de tokens processados, isoladamente, não garante a qualidade do treinamento: também é necessário considerar o tamanho real do modelo, a diversidade dos dados, a taxa de aprendizado e a eficiência computacional.

## 4\. Estratégia de paralelismo

A estratégia adotada foi o **Pipeline Parallelism**, que distribui diferentes partes do modelo entre as GPUs disponíveis.

Nessa abordagem, cada GPU executa uma parte da rede neural, encaminhando as ativações para o próximo estágio do pipeline durante a propagação direta e recebendo os gradientes correspondentes durante a retropropagação.

A escolha dessa estratégia teve como objetivos:

- Explorar a distribuição do modelo entre duas GPUs.
- Experimentar uma arquitetura de treinamento paralelizado.
- Avaliar a execução com recursos computacionais limitados.
- Facilitar testes rápidos de diferentes configurações do modelo.

Uma alternativa seria utilizar **Distributed Data Parallel (DDP)**, no qual cada GPU mantém uma cópia do modelo e processa diferentes partes dos dados. Para modelos pequenos, o DDP pode ser mais eficiente, dependendo do tamanho do lote, da comunicação entre GPUs e dos custos de sincronização.

Assim, o pipeline parallelism foi adotado principalmente como uma estratégia experimental, e sua eficiência deve ser avaliada por métricas como tokens por segundo, utilização das GPUs e tempo total de treinamento.

## 5\. Arquitetura do modelo

A configuração utilizada no experimento foi:

```
VOCAB_SIZE = 50_257
EMBEDDING_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2
BLOCK_SIZE = 450
DROPOUT = 0.0
```

| Hiperparâmetro | Valor | Descrição |
| --- | --- | --- |
| `VOCAB_SIZE` | 50.257 | Tamanho do vocabulário |
| `EMBEDDING_DIM` | 128 | Dimensão das representações vetoriais |
| `NUM_HEADS` | 2 | Número de cabeças de atenção |
| `NUM_LAYERS` | 2 | Número de camadas Transformer |
| `BLOCK_SIZE` | 450 | Comprimento máximo da sequência de contexto |
| `DROPOUT` | 0.0 | Taxa de dropout |

### 5.1. Tokenizador

Foi utilizado o tokenizador do GPT-2, com vocabulário de 50.257 tokens.

A escolha ocorreu pela disponibilidade de um tokenizador previamente treinado, evitando a necessidade de construir e treinar um vocabulário próprio para o experimento.

Entretanto, o tokenizador não implica que o modelo de linguagem tenha sido pré-treinado. O modelo pode utilizar o vocabulário do GPT-2 enquanto aprende seus próprios parâmetros a partir do dataset TinyStories.

### 5.2. Capacidade do modelo

A arquitetura possui apenas duas camadas Transformer, dimensão de embedding de 128 e duas cabeças de atenção.

Embora o vocabulário seja relativamente grande em comparação com a dimensão interna do modelo, isso não significa, por si só, que a maior parte dos parâmetros esteja nos embeddings. A proporção depende também do tamanho da matriz de saída, do compartilhamento de pesos entre embeddings e camada de saída e da implementação dos blocos Transformer.

Por isso, o número total de parâmetros deve ser medido diretamente na implementação, em vez de ser inferido apenas a partir dos hiperparâmetros.

## 6\. Resultados e análise

Ao final de aproximadamente 21 minutos de treinamento, foi registrado um _loss_ final de **2,1948**.

Esse valor fornece uma referência para avaliar o comportamento do modelo durante o treinamento, mas não é suficiente, isoladamente, para determinar sua capacidade de generalização ou afirmar que ele está memorizando os dados.

Para avaliar melhor os resultados, seria necessário acompanhar:

- A curva de _loss_ durante o treinamento.
- O _loss_ de validação em dados não utilizados no treinamento.
- A qualidade das histórias geradas pelo modelo.
- A velocidade de treinamento em tokens por segundo.
- A utilização de memória e o aproveitamento das duas GPUs.

Como o dataset TinyStories contém histórias relativamente curtas e utiliza uma distribuição de linguagem específica, os resultados também devem ser interpretados considerando as características desse conjunto de dados.
tokens antes de concluir o tamanho do modelo.

Também recomendo verificar se os 65 TFLOPs representam a capacidade teórica por GPU, a capacidade agregada das duas GPUs ou um valor de referência ajustado pelo fator de utilização. Essa distinção muda a interpretação do orçamento calculado.

Se quiser, posso ainda preparar uma versão mais enxuta, com um estilo de README de projeto open source, incluindo seções de instalação, execução, requisitos e estrutura do repositório.
