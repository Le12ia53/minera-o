# Projeto único: anomalias em logs sintéticos e reais

Os conjuntos ficam em pastas separadas dentro do mesmo projeto. Execute os dois experimentos com comandos diferentes. Cada execução grava suas próprias métricas e atualiza um artigo IEEEtran de seis páginas que explica a definição de anomalia, detecção, tratamento, origem dos dados, preparação e resultados.

## Preparar o ambiente (uma vez)

```bash
cd PROJETO_ANOMALIAS_REAL_E_SINTETICO
python3 -m venv .venv
source .venv/bin/activate
mkdir -p .tmp-pip
TMPDIR="$PWD/.tmp-pip" python -m pip install --no-cache-dir -r requirements-14.txt
```

Se a instalação apresentar `No space left on device`, o limite pode estar em `/tmp`, mesmo com espaço livre no disco do projeto. O comando acima usa `.tmp-pip` no disco para os arquivos temporários e desativa o cache de downloads. Para conferir espaço e inodes, execute `df -h . /tmp` e `df -i . /tmp`. Após a instalação terminar, a pasta `.tmp-pip` pode ser removida se estiver vazia com `rmdir .tmp-pip`.

Para compilar o PDF automaticamente em Ubuntu/WSL: `sudo apt install texlive-latex-base texlive-publishers`. Sem LaTeX, o projeto ainda escreve o `.tex`. Bibliotecas dos 14 métodos originais podem exigir ajustes no seu Python; cada erro é registrado sem atribuir resultado fictício.

## Comando 1: log sintético da professora (500 casos)

```bash
python executar_sintetico.py
```

A origem é o modelo XML genérico disponibilizado pela professora Sarajane Marques Peres, visualizado no Bizagi Modeler e usado no BIMP. Os MXML originais com 5.000 casos e o BPMN estão em `data/sintetico/fonte_5000/`. Uma amostra reproduzível de 500 casos está em `data/sintetico/amostra_500/`: 475 normais e 25 alterados, com o mesmo ID nos arquivos limpo e anômalo. A amostra já acompanha o pacote.

O comando padrão executa raridade de transições (F1 medido de 0,920 nessa amostra). Para escolher métodos do conjunto de 14 implementações: `python executar_sintetico.py --methods naive sampling likelihood`. Resultados e relatório: `results/professora-500-5pct/`. Para refazer a extração: `python executar_sintetico.py --regenerate`.

## Comando 2: log real BPIC12 usado também em X-Processes

```bash
python executar_real.py --cases 1000 --rate 0.20 --seed 42 --epochs 10
```

O XES completo está em `data/real/BPI_Challenge_2012.xes.gz`. BPIC12 consta da Tabela 6 do artigo X-Processes de Fantinato, Peres e Reijers. O artigo estuda descoberta de modelos; este projeto estuda anomalias. O comando usa 1.000 casos reais e modifica artificialmente 200 (omissão, repetição, troca ou atividade desconhecida). As cópias limpa e alterada em XES.GZ e os rótulos externos já estão em `data/experimento/bpic12-anomalias/`.

Por padrão tenta os 14 métodos. Ensaio curto: `python executar_real.py --dataset teste-real-100 --cases 100 --methods naive sampling likelihood --epochs 1 --timeout 120`. Resultados e relatório: `results/bpic12-anomalias/` ou o nome passado em `--dataset`.

As tabelas antigas de quatro métodos em `results/*.csv` aparecem no relatório como experimento anterior separado; não são resultados dos 14 métodos. F1 avalia alterações artificiais em casos reais, não anomalias naturais validadas por especialistas.

## Relatório atualizado depois de cada comando

`results/<dataset>/RELATORIO_IEEE.pdf` reúne os resultados disponíveis do sintético e do real. A seção dos 14 métodos só atribui números a métodos com `status: ok`; falhas e métodos ainda não executados são identificados. `.tex` e `references.bib` ficam junto ao PDF. Após rodar os dois comandos, o relatório gerado pelo segundo contém os resultados atualizados dos dois experimentos.

Para recompilar sem repetir os métodos: `python gerar_relatorio.py --dataset bpic12-anomalias`.
