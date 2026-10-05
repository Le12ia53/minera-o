"""Gera artigo IEEEtran dos resultados efetivamente calculados (nunca inventa métricas)."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ('likelihood', 'naive', 'sampling', 'dae', 'binetv3', 'binetv2',
         'gama', 'vae', 'lae', 'gae', 'grasped', 'leverage', 'w2vlof', 'vaeocsvm')


def tex(s):
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')


def metric(row, key):
    value = row.get(key)
    return f'{value:.3f}'.replace('.', ',') if isinstance(value, (int, float)) else '--'


def generate(dataset, require_pdf=False):
    directory = ROOT / 'results' / dataset
    summary_path = directory / 'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(f'Execute os métodos antes: {summary_path}')
    current_rows = json.loads(summary_path.read_text(encoding='utf-8'))
    if not isinstance(current_rows, list) or not current_rows:
        raise ValueError('Resumo de métodos vazio')
    real_dataset = dataset if dataset != 'professora-500-5pct' else 'bpic12-anomalias'
    real_summary = ROOT / 'results' / real_dataset / 'summary.json'
    rows = json.loads(real_summary.read_text(encoding='utf-8')) if real_summary.exists() else []
    synthetic_summary = ROOT / 'results/professora-500-5pct/summary.json'
    synthetic_rows = json.loads(synthetic_summary.read_text(encoding='utf-8')) if synthetic_summary.exists() else []
    synthetic_ok = [r for r in synthetic_rows if r.get('status') == 'ok']
    synthetic_lines = ('\n'.join(
        f"{tex(r['method'])} & {metric(r,'precision')} & {metric(r,'recall')} & {metric(r,'f1')} \\\\"
        for r in synthetic_ok
    ) or r'\multicolumn{4}{c}{Sem método sintético concluído.} \\')
    synthetic_results = (
        r'\begin{table}[t]\centering\caption{Resultados da amostra sintética de 500 casos}\begin{tabular}{lrrr}\toprule '
        + 'Método & P & R & F1' + r'\\\midrule ' + synthetic_lines
        + r' \bottomrule\end{tabular}\end{table}'
    )
    synthetic_f1 = ('; '.join(f"{tex(r['method'])}: F1 {metric(r,'f1')}" for r in synthetic_ok)
                    if synthetic_ok else 'sem método sintético concluído')
    older_real = ROOT / 'results/metricas_real_controlado.csv'
    previous = []
    if older_real.exists():
        with older_real.open(encoding='utf-8-sig', newline='') as file:
            previous = list(csv.DictReader(file))
    historical = ''
    if previous:
        previous_rows = '\n'.join(
            f"{tex(r['metodo'])} & {float(r['f1']):.3f} & {float(r['roc_auc']):.3f} \\\\"
            for r in previous
        )
        historical = (r'\begin{table}[t]\centering\caption{Experimento real anterior: quatro baselines, 1.000 casos}\begin{tabular}{lrr}\toprule '
                      + 'Método & F1 & AUC' + r'\\\midrule ' + previous_rows
                      + r' \bottomrule\end{tabular}\end{table}')
    protocol_path = ROOT / 'data' / 'experimento' / real_dataset / 'protocolo.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    cases = protocol.get('n_cases', rows[0].get('n_cases', 'não registrado') if rows else 'não registrado')
    changed = protocol.get('n_modified', rows[0].get('n_anomalies', 'não registrado') if rows else 'não registrado')
    rate = protocol.get('rate')
    percent = f'{100 * rate:g}\\%' if isinstance(rate, (int,float)) else 'não registrada'
    measured = [r for r in rows if r.get('status') == 'ok']
    failed = [r for r in rows if r.get('status') != 'ok']
    table = '\n'.join(
        f"{tex(r['method'])} & {metric(r,'precision')} & {metric(r,'recall')} & {metric(r,'f1')} & {metric(r,'roc_auc')} \\\\"
        for r in measured
    ) or r'\multicolumn{5}{c}{Nenhum método concluído nesta execução.} \\'
    states = '\n'.join(
        f"{tex(method)} & {tex(next((r.get('status','não executado') for r in rows if r.get('method') == method), 'não executado'))} \\\\"
        for method in ORDER
    )
    errors = ('\n'.join(
        f"\\item \\texttt{{{tex(r['method'])}}}: {tex(str(r.get('error', 'erro não registrado')).splitlines()[-1][:155])}."
        for r in failed
    ) if failed else r'\item Nenhuma falha registrada entre os métodos solicitados nesta etapa.')
    types = protocol.get('types', {})
    type_text = ', '.join(f'{tex(k)}: {v}' for k,v in types.items()) or 'distribuição por tipo não registrada'
    body = r'''\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs,amsmath,url}
\title{Detecção de Anomalias em Logs de Eventos: Modelo Sintético da Disciplina e Dados Reais Utilizados em X-Processes}
\author{\IEEEauthorblockN{Vanderleia Matos Costa e colaboradores}\\\IEEEauthorblockA{SIN5025 --- Mineração de Processos\\Universidade de São Paulo}}
\begin{document}
\maketitle
\begin{abstract}
Estudam-se anomalias em duas fontes distintas: uma amostra de 500 casos do log sintético gerado no BIMP a partir do modelo XML da disciplina e o conjunto observacional BPIC12, utilizado pelos autores de X-Processes. No sintético, 25 casos da amostra foram modificados; __SYNTH_F1__. Para o experimento real, __CASES__ casos foram escolhidos e __CHANGED__ foram modificados artificialmente (__RATE__). Os resultados desta edição são lidos dos arquivos summary.json após cada execução. Apenas métodos concluídos recebem métricas; falhas e ausências aparecem no estado da execução. O BPI possui casos reais, mas os rótulos deste experimento indicam modificações controladas, e não anomalias naturais comprovadas.
\end{abstract}
\begin{IEEEkeywords}mineração de processos, detecção de anomalias, XES, dados reais, simulação\end{IEEEkeywords}
\section{Introdução}
Um log de eventos reúne atividades, instante e identificador de caso. Uma sequência cronológica de atividades é um trace. A qualidade dos registros importa para descobrir processos, comparar comportamentos e investigar exceções. Um caso incomum não é necessariamente incorreto: escolhas válidas, paralelismo e mudanças do processo podem gerar sequências raras \cite{vanderaalst2016process,vanderaalstcarmona2022handbook}.

O objetivo é explicar o que se chama anomalia, quais desvios podem ser detectados e tratados e como comparar resultados em fontes sintética e observacional. Os autores Fantinato, Peres e Reijers aplicaram X-Processes a 12 logs reais para descobrir modelos \cite{fantinato2023xprocesses}; aqui se estuda detecção de alterações em um dos mesmos conjuntos, BPIC12. Não são reproduzidos os resultados de qualidade de descoberta como se fossem métricas de anomalias.

\section{Definição e identificação}
Seja $\sigma=\langle a_1,\ldots,a_m\rangle$ um trace. Neste experimento rotulado, anomalia significa uma edição deliberada em uma cópia de $\sigma$, com tipo e identificador registrados. Essa definição operacional permite medir acerto comparando a observação com a cópia anterior. Em produção, atividade rara, atividade inexistente, repetição indevida, omissão e troca de ordem são apenas indícios; confirmar anomalia requer conhecer regras, exceções e contexto \cite{suriadi2017patterns}.

\subsection{Estratégias de detecção}
Modelos normativos e verificações de conformidade permitem confrontar o trace com comportamento permitido; escores de raridade de transições comparam pares consecutivos; LOF estima densidade relativa; Isolation Forest procura casos isolados; redes neurais podem aprender representações de eventos. Um limiar ou orçamento de alertas converte ranking em suspeitas. O BPMN do experimento sintético serve à simulação, sem ser inserido como rótulo no método estatístico \cite{breunig2000lof,liu2008isolation}.

\clearpage
\section{Quais anomalias podem ser tratadas?}
Omissão de atividade, repetição de registro, permuta de atividades e inserção de atividade desconhecida são alterações estruturais controláveis por edição da sequência. Também existem desvios de tempo, recurso, atributo de dado e relação entre objetos: como o programa desta experiência utiliza somente atividades por caso, não se deve alegar que essas últimas categorias foram medidas. Campos de tempo e recurso podem ser analisados em trabalhos posteriores.

Há alternativas de tratamento: preservar e encaminhar à revisão, corrigir um erro confirmado, adicionar evento faltante com evidência externa, remover duplicata comprovada, normalizar códigos de atividade, ou atualizar o modelo se o comportamento raro for legítimo. Uma sugestão de reparo baseada em protótipos observados não equivale à restauração comprovada sem comparação com referência limpa. Deve-se registrar original, justificativa, edição e responsável \cite{weske2024bpm,suriadi2017patterns}.

\section{Dados sintéticos da professora}
A professora Sarajane Marques Peres disponibilizou ``Modelo de processo (fonte / XML genérico) --- Para ser usado no simulador''. A visualização no Bizagi Modeler mostrou 26 atividades de A até Z, decisões XOR, ramificações paralelas, reconvergências, um retorno de fluxo e mais de uma possibilidade de término. A versão BPMN exportada acompanha o projeto sintético. O modelo não foi substituído por um fluxo próprio mais simples.

Após configurar o processo no BIMP, incluindo durações, recursos e probabilidades das saídas XOR, foram gerados 5.000 casos em MXML compactado. Preservou-se o log limpo e, em uma cópia, um script Python modificou 250 casos (5\%), com semente 2026. O projeto extrai uma amostra controlada de 500 desses casos: 25 rotulados como alterados e 475 não alterados, mantendo os mesmos IDs nos arquivos limpo e anômalo. Os tipos foram inserção final de A, omissão de A, repetição de A e substituição de A por Z. Os rótulos são externos ao arquivo pontuado. Não se infere do log a configuração exata de todas as probabilidades usadas no simulador.

\section{Resultado sintético registrado}
O código calcula escores e confronta alertas com o arquivo externo de rótulos da amostra. A Tabela I é atualizada depois da execução sintética. São comparados 500 casos, sendo 25 alterados; uma tabela vazia significa que ainda não há execução concluída, e não que o resultado seja zero. Os arquivos de caso preservam a pontuação e a decisão para auditoria. __SYNTH_RESULTS__

\clearpage
\section{Origem dos dados reais}
O BPIC12, do BPI Challenge 2012, é um log observacional de solicitações de empréstimo disponibilizado por van Dongen no repositório 4TU \cite{vandongen2012bpi}. A Tabela 6 do artigo X-Processes \cite{fantinato2023xprocesses} inclui BPIC12 entre os 12 logs de referência e reporta 13.087 casos, 262.200 eventos, 24 atividades e 4.366 variantes; o arquivo XES.GZ incluído neste projeto tem esses totais. O repositório X-Processes também contém um arquivo denominado test.xes.gz; esse arquivo menor não é o BPIC12 usado aqui. A coincidência de nome e estatísticas identifica o conjunto de origem; a contribuição experimental desta pesquisa é uma tarefa nova de detecção.

\section{Protocolo XES e prevenção de vazamento}
O comando preparar\_xes.py lê o XES original sem modificá-lo, seleciona casos elegíveis de forma determinística e grava uma amostra limpa em XES.GZ. Sua cópia modificada recebe uma das quatro operações: omissão de evento interno, repetição de evento, troca dos nomes das duas primeiras atividades e inserção de atividade \texttt{\_\_UNKNOWN\_\_}. A tabela externa identifica os casos alterados e preserva suas sequências antes/depois. Nessa execução: __CASES__ casos, __CHANGED__ modificações, taxa __RATE__, distribuição __TYPES__.

O importador converte o XES alterado para a estrutura esperada pelos 14 métodos BPAD, em que todos os exemplos são marcados como normais na entrada. O arquivo de referência é aberto pelo programa apenas para avaliar pontuações e sugestões. O protocolo compara traces reais aos mesmos traces após ruído artificial. A organização original não forneceu uma lista de anomalias naturais para esses casos, logo métricas de precisão e F1 não devem ser interpretadas como validação de fraude ou irregularidade real.

\section{Métricas e reparação}
Após fixar o orçamento de alertas, $P=TP/(TP+FP)$, $R=TP/(TP+FN)$ e $F_1=2PR/(P+R)$. AP e AUC-ROC avaliam a ordenação por escore. Uma correção proposta pode ser contada como exata apenas se igualar a cópia limpa do trace, evento por evento na perspectiva escolhida; a contagem de sugestões aplicadas, por si só, não comprova recuperação. O erro do método deve permanecer visível quando ocorrer.

\clearpage
\section{Resultados gerados para o log real}
__HISTORICAL__
O projeto também preserva um experimento anterior com quatro métodos de referência em 1.000 casos BPIC12, com 200 casos corrompidos artificialmente; suas métricas são mostradas separadamente. A tabela dos 14 métodos é construída com entradas cujo status é ok em results/__REAL_DATASET__/summary.json. Métodos não concluídos não recebem valores substitutos. Nenhum desses números é atribuído ao artigo X-Processes.

\begin{table}[t]\centering\caption{Resultados reais controlados; somente execuções concluídas}\begin{tabular}{lrrrr}\toprule
Método & P & R & F1 & AUC\\\midrule
__TABLE__
\bottomrule\end{tabular}\end{table}

\begin{table}[t]\centering\caption{Estado dos 14 métodos disponibilizados}\begin{tabular}{ll}\toprule
Método & Estado\\\midrule
__STATES__
\bottomrule\end{tabular}\end{table}

\section{Leitura responsável dos resultados}
Mesmo que todos os detectores executem, a proporção de alertas e a taxa de corrupção determinam parte das métricas observadas. Como a comparação utiliza apenas atividades e uma edição por caso, o teste não mede anomalias de recurso ou tempo. Um resultado de reparação é auditável consultando os arquivos de escores, casos e sugestões no mesmo diretório. Para comparar taxas de ruído, devem-se repetir execuções com sementes e parâmetros explícitos; somente então se pode calcular intervalos de variação.

\clearpage
\section{Falhas, limites e interpretação}
Cada implementação legada é executada isoladamente. Se faltarem dependências ou houver incompatibilidades de versão, o método recebe status failed e erro em summary.json. A tabela abaixo registra as falhas desta execução, sem convertê-las em sucessos.
\begin{itemize}
__ERRORS__
\end{itemize}

O X-Processes otimiza modelos descobertos, com foco em adequação, precisão, generalização e simplicidade \cite{fantinato2023xprocesses}. Essas medidas não são sinônimas de precisão, revocação e F1 de um detector de casos alterados. O log real pode conter comportamentos raros perfeitamente legítimos; sem validação de especialistas, o protocolo só estabelece capacidade de recuperar mudanças que ele mesmo introduziu. A amostra também não prova transferência de um método treinado sobre o modelo da professora para processos financeiros.

\section{Reprodutibilidade}
Este pacote mantém os logs em data/sintetico e data/real, sob um mesmo diretório de projeto. Execute executar\_sintetico.py para pontuar os 500 casos MXML ou executar\_real.py para os casos XES. Cada comando gera um resumo de métodos e recompila este relatório a partir dos resultados presentes. Confira rotulos.csv externo, escores por caso e status antes de usar as tabelas. O PDF depende de pdfLaTeX e BibTeX; o arquivo .tex é sempre escrito. Não é necessário instalar X-Processes para executar BPAD: a referência ao estudo identifica o conjunto real escolhido.

\clearpage
\section{Conclusão}
O modelo da disciplina, iniciado no XML fornecido e executado no BIMP, permite avaliação controlada porque o log limpo foi preservado. Na amostra de 500 casos, __SYNTH_F1__. O BPIC12 fornece casos observacionais empregados também pelos pesquisadores de X-Processes; aqui uma cópia recebe alterações controladas para permitir medir detecção e estudar sugestões de reparo. A versão real contém __SUCCESS__ método(s) concluído(s) entre __REQUESTED__ solicitados; as tabelas descrevem apenas números gravados. Sem rótulos humanos de anomalias naturais, recomenda-se revisão humana e repetições com diferentes sementes e taxas.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
'''
    replacements = {
        '__CASES__': str(cases), '__CHANGED__': str(changed),
        '__RATE__': percent, '__TYPES__': type_text, '__DATASET__': tex(dataset),
        '__TABLE__': table, '__STATES__': states, '__ERRORS__': errors,
        '__SYNTH_RESULTS__': synthetic_results, '__SYNTH_F1__': synthetic_f1,
        '__HISTORICAL__': historical,
        '__REAL_DATASET__': tex(real_dataset),
        '__SUCCESS__': str(len(measured)), '__REQUESTED__': str(len(rows))
    }
    for key, value in replacements.items():
        body = body.replace(key, value)
    tex_file = directory / 'RELATORIO_IEEE.tex'
    tex_file.write_text(body, encoding='utf-8')
    shutil.copy2(ROOT / 'references.bib', directory / 'references.bib')
    shutil.copy2(ROOT / 'IEEEtran.bst', directory / 'IEEEtran.bst')
    print('Fonte:', tex_file)
    if not shutil.which('pdflatex') or not shutil.which('bibtex'):
        if require_pdf:
            raise RuntimeError('Instale texlive-latex-base e BibTeX para obter PDF; o .tex já foi gerado.')
        print('PDF pendente: pdflatex e bibtex não estão instalados; use o .tex gerado.')
        return tex_file
    command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
               f'-output-directory={directory}', str(tex_file)]
    for cmd in (command, ['bibtex', tex_file.stem], command, command):
        process = subprocess.run(cmd, cwd=directory if cmd[0] == 'bibtex' else ROOT,
                                 capture_output=True, text=True, errors='replace')
        if process.returncode:
            raise RuntimeError(f'Falha ao compilar relatório: {process.stdout[-1300:]} {process.stderr[-300:]}')
    pdf = directory / 'RELATORIO_IEEE.pdf'
    print('PDF:', pdf)
    return pdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--require-pdf', action='store_true')
    args = parser.parse_args()
    generate(args.dataset, args.require_pdf)
