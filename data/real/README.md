# Log real — BPI Challenge 2012

O projeto usa o **BPI Challenge 2012**, uma das categorias de logs reais
expressamente permitidas no syllabus da disciplina.

## Fonte e licença

- Autor: Boudewijn F. van Dongen.
- Título: BPI Challenge 2012.
- Fonte oficial: https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204
- DOI: https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- Licença: CC0.
- Processo: solicitação de crédito de uma instituição financeira neerlandesa.

O arquivo incluído, BPI_Challenge_2012.xes.gz, é a versão XES completa
compactada. Estatísticas verificadas pelo código:

| Medida | Valor |
|---|---:|
| Casos | 13.087 |
| Eventos | 262.200 |
| Atividades | 24 |
| Variantes | 4.366 |
| Menor trace | 3 eventos |
| Maior trace | 175 eventos |

SHA-256 do arquivo incluído:

e7b901fbc35cb3995196fe87696090b724edfb7a00a11cbfd5086624fccf39cb

## Executar

Na raiz do projeto:

~~~bash
python run_real_log.py
~~~

Configuração padrão:

- amostra controlada: 1.000 casos;
- casos alterados: 20%;
- semente: 42;
- ranking exploratório: 5% dos maiores escores.

## Duas avaliações diferentes

### 1. Corrupção controlada em casos reais

Uma cópia de 1.000 traces reais é preservada. Em 200 casos são injetados
ausência, duplicação, troca ou atividade desconhecida. Como o trace anterior à
alteração é conhecido, podem ser calculados precisão, recall, F1 e recuperação
exata.

### 2. Raridade no log real original

O log completo, sem alterações, recebe um escore baseado na raridade de
transições. Os 654 maiores escores formam uma fila de inspeção. Eles não são
tratados como erros verdadeiros e não recebem métricas supervisionadas.

## Arquivos produzidos

- results/estatisticas_log_real.csv;
- results/ranking_log_real.csv;
- results/metricas_real_controlado.csv;
- results/metricas_reparo_real_controlado.csv;
- results/metricas_real_por_tipo_ruido.csv;
- results/predicoes_real_controlado.csv;
- data/real/amostra_bpi2012_ruidosa.csv;
- data/real/rotulos_amostra_bpi2012.csv.
